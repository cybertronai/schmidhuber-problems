"""Single predeclared29x29, seed0,800-epoch online MLP arm.

Requires a separately reviewed operator watchdog/lease; this module does not
authorize a device run. It never downloads data. Final test entry is exclusive
and occurs only after all800 original-training validations select the model.
"""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import platform
import re
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
from paper_online import PaperConfig, epoch_order
from paper_cuda import OnlineMLP, augment, configure_cuda, load_checkpoint, save_checkpoint
from paper_inputs import read_train, digest
from paper_pilot import source_digest
from paper_runstate import RunStore, after_validation, atomic_json, canonical, file_sha, initial_state, read_json, regular, value_sha


class ControlledStop(RuntimeError): pass


def protocol():
    return {'schemaVersion':1,'armId':'paper29-seed0','epochs':800,'trainExamples':60000,
            'updates':48000000,'side':29,'seed':0,'graphChunk':32,
            'checkpointThreshold':6000,'checkpointRule':'First completed32-example chunk at/after each6000-update threshold, plus pre/post validation and controlled stop',
            'retention':'latest + recoverable previous + best; at most3 committed generations, at most4 while publishing; interrupted partials preserved within512MiB bound',
            'selection':'Minimum integer mistakes on all60000 original undeformed training examples at each completed epoch; earliest epoch on ties',
            'finalTest':'Exactly one reserved10000-example evaluation of selected best after800 validations; interrupted reservation cannot be repeated',
            'initialPrefix':'Admitted4096-update seed0 pilot may be continued without repeated/discarded counted updates',
            'precision':'FP32, deterministic, noTF32; same sequential GEMV update kernels as admitted pilot',
            'validation':'Read-only captured sequential GEMV forwards, same32 input grouping; eager/captured full-size output equality before training continues'}


def driver_digest():
    root=Path(__file__).resolve().parent
    return value_sha({n:digest(root/n) for n in ('paper_train.py','paper_runstate.py')})


class EvaluationGraph:
    """Read-only forward graph, no batch reduction or parameter updates."""
    def __init__(self,model,count):
        if count not in (16,32): raise ValueError('Literal finite evaluation chunks')
        self.model=model; self.t=model.torch; self.count=count; t=self.t
        self.x=t.zeros((count,model.sizes[0]),device='cuda',dtype=t.float32)
        self.out=t.empty((count,model.sizes[-1]),device='cuda',dtype=t.float32)
        before=[v.clone() for v in [*model.weights,*model.biases]]
        stream=t.cuda.Stream(); stream.wait_stream(t.cuda.current_stream())
        with t.no_grad(),t.cuda.stream(stream):
            for _ in range(3): model._forward(self.x[0])
        t.cuda.current_stream().wait_stream(stream)
        self.graph=t.cuda.CUDAGraph()
        with t.no_grad(),t.cuda.graph(self.graph):
            for i in range(count):
                model._forward(self.x[i]); self.out[i].copy_(model.acts[-1])
        t.cuda.synchronize()
        if not all(t.equal(v,old) for v,old in zip([*model.weights,*model.biases],before)):
            raise ValueError('Read-only evaluation capture changed parameters')

    def predict(self,images):
        if len(images)!=self.count: raise ValueError('Exact evaluated example count')
        self.x.copy_(images.reshape(self.count,-1).mul(2).sub(1))
        self.graph.replay()
        return self.out

    def control(self,images):
        actual=self.predict(images).clone(); expected=[]
        with self.t.no_grad():
            for image in images:
                self.model._forward(image.reshape(-1).mul(2).sub(1)); expected.append(self.model.acts[-1].clone())
        self.t.cuda.synchronize()
        if not self.t.equal(actual,self.t.stack(expected)) or not self.t.isfinite(actual).all().item():
            raise ValueError('Full-size captured/eager read-only validation differs')


def evaluate(images,labels,graphs,check):
    if len(images)!=len(labels) or len(images)not in (60000,10000): raise ValueError('Exact declared evaluation split')
    started=time.monotonic(); errors=0
    for first in range(0,len(images),32):
        check(); n=min(32,len(images)-first); out=graphs[n].predict(images[first:first+n])
        if not graphs[n].t.isfinite(out).all().item(): raise ValueError('Nonfinite actual evaluation output')
        errors+=int((out.argmax(dim=1)!=labels[first:first+n]).sum().item())
    return {'errors':errors,'examples':len(images),'seconds':time.monotonic()-started,
            'method':'captured sequential FP32 GEMV, read-only weights; exactly one prediction per example'}


def read_test_once(root,manifest_sha,store,reservation):
    # A successfully created exclusive reservation is required before any test
    # image/label file is opened. No exploratory test metrics or second pass.
    if read_json(store.root/'FINAL_TEST.json')!=reservation or reservation.get('state')!='reserved':
        raise ValueError('Genuine original one-use test reservation required')
    root=Path(root)
    if digest(root/'MANIFEST.json')!=manifest_sha: raise ValueError('Final test identity manifest differs')
    manifest=json.loads((root/'MANIFEST.json').read_text(encoding='utf8')); split=manifest['splits']['test']
    if split['count']!=10000: raise ValueError('Original test count')
    values=[]
    for key,size in [('images',7840000),('labels',10000)]:
        row=split[key]; p=root/row['path']
        if row['path']!='test.'+key or row['size']!=size or regular(p).st_size!=size or digest(p)!=row['sha256']:
            raise ValueError('Exact authenticated original test stream required')
        values.append(bytearray(p.read_bytes()))
    if any(v>9 for v in values[1]): raise ValueError('Test label range')
    return values


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True); parser.add_argument('--data-manifest-sha256',required=True)
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--stop-file',type=Path,required=True)
    parser.add_argument('--seconds',type=int,required=True); parser.add_argument('--resume',choices=['pilot','latest'],required=True)
    parser.add_argument('--pilot-checkpoint',type=Path); parser.add_argument('--final-test-approved',action='store_true')
    parser.add_argument('--initial-prefix-commit-sha256',required=True)
    args=parser.parse_args(); cfg=PaperConfig(); cfg.validate()
    if not 1<=args.seconds<=27000: raise ValueError('Finite sustained27000s work budget required')
    if not re.fullmatch('[0-9a-f]{64}',args.initial_prefix_commit_sha256): raise ValueError('Exact admitted prefix commit identity required')
    start=time.monotonic()
    def check():
        if args.stop_file.exists() or time.monotonic()-start>=args.seconds: raise ControlledStop('Operator stop/finite work budget')
    check(); images,labels=read_train(args.data,args.data_manifest_sha256)
    import torch
    runtime=configure_cuda(torch); runtime['python']=platform.python_version()
    binding={'configSHA256':cfg.digest(),'sourceSHA256':source_digest(),'dataSHA256':args.data_manifest_sha256,
             'runtime':runtime,'rngScheme':'init=CPUtorch(seed);shuffle=PythonRandom((seed<<32)+epoch);augmentation=CUDAgenerator(seed+1000003*epoch+block)'}
    arm={'protocol':protocol(),'configSHA256':cfg.digest(),'scientificSourceSHA256':source_digest(),
         'driverSHA256':driver_digest(),'dataSHA256':args.data_manifest_sha256,'runtime':runtime,
         'initialPrefixCommitSHA256':args.initial_prefix_commit_sha256}
    store=RunStore(args.root,arm)
    if (store.root/'FINAL_TEST.json').exists():
        raise ValueError('Final test already reserved/consumed; no repeated inference or training')
    binding.update(sustainedArmSHA256=store.binding,sustainedDriverSHA256=driver_digest())
    @lru_cache(maxsize=2)
    def order_hash(epoch): return value_sha(epoch_order(cfg.seed,epoch))
    def checkpoint_binding(s):
        return {**binding,'learningRate':cfg.learning_rate(min(s['epoch'],799)),
                'epochOrderSHA256':order_hash(s['epoch']),
                'mutableRNGState':None,'rngExplanation':'Private fixed seed/epoch/block RNG addressing; no draws between update checkpoints.'}
    model=OnlineMLP(cfg.sizes,cfg.seed,'cuda')
    if args.resume=='pilot':
        if args.pilot_checkpoint is None or (store.root/'STATE.json').exists(): raise ValueError('Only one initial pilot migration')
        if file_sha(args.pilot_checkpoint/'COMMIT.json')!=args.initial_prefix_commit_sha256: raise ValueError('Admitted actual pilot prefix was substituted')
        expected={k:v for k,v in binding.items() if k not in ('sustainedArmSHA256','sustainedDriverSHA256')}
        meta=load_checkpoint(model,args.pilot_checkpoint,expected)
        if tuple(meta.get(k) for k in ('epoch','cursor','updates','testEvaluations','validationPending'))!=(0,4096,4096,0,True):
            raise ValueError('Exact admitted4096-update pilot prefix required')
        state=initial_state(4096); state=store.publish(model,state,checkpoint_binding(state),save_checkpoint)
    else:
        if args.pilot_checkpoint is not None: raise ValueError('Resume mode cannot substitute original pilot')
        state=store.state();pending=store.pending()
        if pending is not None:
            # Completion is established by actual committed bytes and native
            # tensor validation, not by an expired writer or a stale lock.
            verified=load_checkpoint(model,pending['checkpoint'],checkpoint_binding(pending['state']))
            state=store.promote_pending(pending,verified)
        proof=store.inspect(state['latest']);meta=load_checkpoint(model,proof['checkpoint'],binding)
        if meta.get('runState')!=state or any(meta.get(k)!=v for k,v in checkpoint_binding(state).items()):
            raise ValueError('Actual resumed model/cursor/LR/RNG addressing differs')
    model.capture(); check()
    import torch.nn.functional as f
    original=f.interpolate(torch.frombuffer(images,dtype=torch.uint8).reshape(60000,1,28,28).to('cuda',torch.float32).div_(255),size=(29,29),mode='bilinear',align_corners=True)
    train_labels=torch.frombuffer(labels,dtype=torch.uint8).to('cuda',torch.int64)
    del images,labels
    target=torch.full((60000,10),-1.,device='cuda'); target.scatter_(1,train_labels[:,None],1.)
    graphs={n:EvaluationGraph(model,n) for n in (32,16)}
    for n,graph in graphs.items(): graph.control(original[:n])
    atomic_json(store.root/'EVALUATION_CONTROL.json',{'passed':True,'chunks':[32,16],'fullSize':True,'capturedEqualsEagerBitwise':True,'parametersUnchanged':True,'testExamplesRead':0})
    timing={'deformationSeconds':0.,'trainingSeconds':0.,'validationSeconds':0.,'checkpointSeconds':0.,'resumedUpdates':state['updates'],
            'updatesThisInvocation':0,'validationCallsThisInvocation':0,'reusedValidationRecords':0,
            'scope':'Only this invocation; prior pilot/partial runtime receipts retained separately; no synthesized total'}
    def snapshot(s):
        if not model.finite(): raise ValueError('Nonfinite original model parameters')
        s={**s,'nextCheckpointAt':((s['updates']//6000)+1)*6000}
        t=time.monotonic(); out=store.publish(model,s,checkpoint_binding(s),save_checkpoint); timing['checkpointSeconds']+=time.monotonic()-t
        atomic_json(store.root/'LIVE_PROGRESS.json',{'state':out,'timing':timing,'elapsedSeconds':time.monotonic()-start,'testEvaluations':0})
        return out
    try:
        while state['epoch']<800 or state['validationPending']:
            check()
            if state['validationPending']:
                result=store.validation(state['epoch']-1,state['latest'],lambda:evaluate(original,train_labels,graphs,check))
                if result['reusedFromJournal']: timing['reusedValidationRecords']+=1
                else:
                    timing['validationSeconds']+=result['seconds']; timing['validationCallsThisInvocation']+=1
                state=snapshot(after_validation(state,result['errors']))
                print(json.dumps({'validatedEpochs':state['validatedEpochs'],'updates':state['updates'],'validationErrors':result['errors'],'bestEpoch':state['bestEpoch'],'bestErrors':state['bestErrors'],'elapsedSeconds':time.monotonic()-start}),flush=True)
                continue
            epoch=state['epoch']; order=torch.tensor(epoch_order(cfg.seed,epoch),device='cuda',dtype=torch.int64)
            t=time.monotonic(); augmented=torch.empty_like(original)
            for block,first in enumerate(range(0,60000,256)):
                check(); augmented[first:first+256]=augment(torch,original[first:first+256],train_labels[first:first+256],cfg.seed+1000003*epoch+block)
            augmented.mul_(2).sub_(1); torch.cuda.synchronize(); timing['deformationSeconds']+=time.monotonic()-t
            while state['epoch']==epoch:
                check(); t=time.monotonic(); cursor=state['cursor']; selected=order[cursor:cursor+32]
                model.replay(augmented[selected].reshape(32,-1),target[selected],cfg.learning_rate(epoch)); torch.cuda.synchronize()
                state={**state,'cursor':cursor+32,'updates':state['updates']+32}; timing['trainingSeconds']+=time.monotonic()-t
                timing['updatesThisInvocation']+=32
                if state['cursor']==60000: state.update(epoch=epoch+1,cursor=0,validationPending=True)
                if state['updates']>=state['nextCheckpointAt'] or state['validationPending']: state=snapshot(state)
            del augmented,order
        if state['validatedEpochs']!=800 or state['updates']!=48000000: raise ValueError('Full original arm incomplete')
        certificate=store.certify_complete(state)
        atomic_json(store.root/'TRAINING_COMPLETE.json',{'completeTraining':True,'epochs':800,'updates':48000000,
                    'validatedEpochs':800,'bestEpoch':state['bestEpoch'],'bestErrors':state['bestErrors'],
                    'armSHA256':store.binding,'testEvaluations':0,'awaitingFinalTest':not args.final_test_approved,
                    'timing':timing,'elapsedSeconds':time.monotonic()-start,'certificate':certificate})
        if not args.final_test_approved:
            print(json.dumps(read_json(store.root/'TRAINING_COMPLETE.json'),sort_keys=True),flush=True)
            return
        reservation=store.reserve_test(state,args.final_test_approved)
        best=store.inspect(state['best']); load_checkpoint(model,best['checkpoint'],binding)
        test_images,test_labels=read_test_once(args.data,args.data_manifest_sha256,store,reservation)
        test=f.interpolate(torch.frombuffer(test_images,dtype=torch.uint8).reshape(10000,1,28,28).to('cuda',torch.float32).div_(255),size=(29,29),mode='bilinear',align_corners=True)
        test_y=torch.frombuffer(test_labels,dtype=torch.uint8).to('cuda',torch.int64)
        final=store.finish_test(reservation,evaluate(test,test_y,graphs,check))
        result={'completePaperRun':True,'epochs':800,'updates':48000000,'side':29,'parameters':cfg.parameters,
                'armSHA256':store.binding,'selection':protocol()['selection'],'bestEpoch':state['bestEpoch'],
                'bestValidationErrors':state['bestErrors'],'testErrors':final['observation']['errors'],'testExamples':10000,'testEvaluations':1,
                'timing':timing,'elapsedSeconds':time.monotonic()-start,'peakAllocatedBytes':torch.cuda.max_memory_allocated(),
                'peakReservedBytes':torch.cuda.max_memory_reserved(),'paperComparisonErrors':35,'originalRepositoryErrors':117,
                'qualification':'Single selected29x29 seed0 arm under the recorded protocol approximations; not an exact recovery of unspecified original implementation details.'}
        atomic_json(store.root/'RESULT.json',result); print(json.dumps(result,sort_keys=True),flush=True)
    except ControlledStop as error:
        # A live parent hold may instead terminate the owned Job immediately;
        # its last already committed chunk/cursor remains the restart authority.
        if state.get('updates')!=store.state()['updates']: state=snapshot(state)
        atomic_json(store.root/'PAUSED.json',{'reason':str(error),'state':state,'timing':timing,'elapsedSeconds':time.monotonic()-start})
        raise SystemExit(75)


if __name__=='__main__': main()
