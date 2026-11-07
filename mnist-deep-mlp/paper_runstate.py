"""Finite single-arm run journal, bounded owned snapshots and one-use final test.

No Torch, downloads, training or device work on import. The operator must supply
the separately reviewed authority/resource supervisor for any actual run.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile

from paper_cuda import _publish_owned


STORE_LIMIT = 512 * 1024**2
SNAPSHOT_ALLOWANCE = 64 * 1024**2
GENERATION = re.compile(r'g[0-9]{8}')


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def value_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    def pairs(items):
        out={}
        for k,v in items:
            if k in out: raise ValueError('Duplicate journal key')
            out[k]=v
        return out
    p=Path(path); regular(p)
    if p.stat().st_size>1024**2: raise ValueError('Bounded run journal required')
    return json.loads(p.read_bytes(),object_pairs_hook=pairs,
                      parse_constant=lambda _:(_ for _ in ()).throw(ValueError('Nonfinite journal')))


def plain(path):
    path=Path(path)
    for p in (path,*path.parents):
        s=p.lstat()
        if stat.S_ISLNK(s.st_mode) or getattr(s,'st_file_attributes',0)&0x400:
            raise ValueError('Run path must not contain links/reparse points')
    return path


def regular(path):
    path=plain(path); s=path.stat()
    if not stat.S_ISREG(s.st_mode) or s.st_nlink!=1:
        raise ValueError('Unique regular journal/snapshot required')
    return s


def file_sha(path):
    regular(path)
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def atomic_json(path,value):
    path=Path(path); plain(path.parent)
    if path.exists(): regular(path)
    descriptor,temp=tempfile.mkstemp(prefix=path.name+'.partial-',dir=path.parent)
    temporary=Path(temp)
    try:
        with os.fdopen(descriptor,'wb') as f:
            f.write(canonical(value)); f.flush(); os.fsync(f.fileno())
        # Same owned-temp-only finite Windows5/32 retry already validated with
        # real tensor pointer publication; no stale reclaim/permission changes.
        _publish_owned(temporary,path,replace=True)
    finally:
        if temporary.exists(): temporary.unlink()


def create_json(path,value):
    path=Path(path); plain(path.parent)
    with path.open('xb') as f:
        f.write(canonical(value)); f.flush(); os.fsync(f.fileno())


def append_json(path,value):
    path=Path(path)
    if path.exists(): regular(path)
    with path.open('ab') as f:
        f.write(canonical(value)+b'\n'); f.flush(); os.fsync(f.fileno())


def initial_state(cursor):
    if type(cursor)is not int or not 0<=cursor<60000 or cursor%32:
        raise ValueError('Original training-prefix cursor required')
    return {'sequence':0,'latest':None,'previous':None,'best':None,
            'bestErrors':None,'bestEpoch':None,'epoch':0,'cursor':cursor,'updates':cursor,
            'nextCheckpointAt':((cursor//6000)+1)*6000,'validatedEpochs':0,'validationPending':False}


def validate_state(state):
    for key in ('sequence','epoch','cursor','updates','nextCheckpointAt','validatedEpochs'):
        if type(state.get(key))is not int: raise ValueError('Exact integer run state')
    if type(state.get('validationPending'))is not bool: raise ValueError('Explicit validation-boundary state required')
    if (not 0<=state['sequence']<20000 or not 0<=state['epoch']<=800 or not 0<=state['cursor']<60000 or state['cursor']%32
        or state['updates']!=state['epoch']*60000+state['cursor']
        or state['validatedEpochs']!=state['epoch']-int(state['validationPending'])
        or state['validationPending'] and (state['cursor']!=0 or state['epoch']==0)
        or state['epoch']==800 and state['cursor']!=0):
        raise ValueError('Run cursor/validation count differs')
    if state['nextCheckpointAt']!=((state['updates']//6000)+1)*6000:
        raise ValueError('Exact next checkpoint threshold required')
    for key in ('latest','previous','best'):
        v=state.get(key)
        if v is not None and (not isinstance(v,str) or not GENERATION.fullmatch(v)):
            raise ValueError('Literal owned snapshot generation required')
    if state['latest']!=(f"g{state['sequence']:08d}" if state['sequence'] else None):
        raise ValueError('Published generation must match exact transaction sequence')
    if state.get('best')is None and (state.get('bestErrors')is not None or state.get('bestEpoch')is not None):
        raise ValueError('Unselected model must not carry a selection statistic')
    if state.get('best') is not None and (type(state.get('bestErrors'))is not int or not 0<=state['bestErrors']<=60000
        or type(state.get('bestEpoch'))is not int or not 0<=state['bestEpoch']<state['validatedEpochs']):
        raise ValueError('Only completed validation can select a best snapshot')
    return state


def after_validation(state,errors):
    validate_state(state)
    if not state['validationPending'] or state['latest']is None or type(errors)is not int or not 0<=errors<=60000:
        raise ValueError('Exactly60000 undeformed training predictions required')
    out=dict(state)
    # Earliest epoch wins equal integer error counts. No test statistic enters.
    if out['bestErrors'] is None or errors<out['bestErrors']:
        out.update(best=out['latest'],bestErrors=errors,bestEpoch=out['epoch']-1)
    out.update(validatedEpochs=out['epoch'],validationPending=False)
    return validate_state(out)


class RunStore:
    def __init__(self,root,arm):
        self.root=Path(root); self.arm=arm; self.binding=value_sha(arm)
        if not self.root.exists(): self.root.mkdir(parents=False)
        plain(self.root)
        path=self.root/'ARM.json'
        if path.exists():
            if read_json(path)!=arm: raise ValueError('Arm source/config/data/protocol differs')
        else:
            # A new arm never adopts pre-existing user files or another arm.
            if list(self.root.iterdir()): raise ValueError('New arm requires an empty owned root')
            create_json(path,arm)
        for name in ('snapshots',):
            p=self.root/name
            if not p.exists(): p.mkdir()
            plain(p)

    def generation(self,name):
        if not isinstance(name,str) or not GENERATION.fullmatch(name): raise ValueError('Exact generation name')
        p=self.root/'snapshots'/name
        if not p.resolve().is_relative_to((self.root/'snapshots').resolve()): raise ValueError('Snapshot escapes owned root')
        return p

    def footprint(self):
        total=0
        for p in self.root.rglob('*'):
            plain(p)
            if p.is_file(): total+=regular(p).st_size
        return total

    def inspect(self,name):
        p=self.generation(name); plain(p)
        owner=read_json(p/'OWNED.json')
        if owner!={'schemaVersion':1,'generation':name,'armSHA256':self.binding}:
            raise ValueError('Snapshot ownership differs')
        pointer=read_json(p/'LATEST.json'); child=pointer.get('checkpoint')
        if not isinstance(child,str) or not re.fullmatch(r'checkpoint-[0-9]{9}',child):
            raise ValueError('Exact inner committed checkpoint required')
        checkpoint=p/child; plain(checkpoint); record=read_json(checkpoint/'COMMIT.json')
        if record.get('complete')is not True or record.get('schemaVersion')!=1:
            raise ValueError('Complete original checkpoint required')
        payload=checkpoint/'tensors.pt'
        if pointer.get('commitSHA256')!=file_sha(checkpoint/'COMMIT.json'):
            raise ValueError('Inner checkpoint pointer digest differs')
        if regular(payload).st_size!=record['payloadBytes'] or file_sha(payload)!=record['payloadSHA256']:
            raise ValueError('Snapshot closed payload differs')
        if set(p.iterdir())!={p/'OWNED.json',p/'LATEST.json',checkpoint} or set(checkpoint.iterdir())!={checkpoint/'COMMIT.json',payload}:
            raise ValueError('Unknown/partial snapshot contents are preserved, never recursively removed')
        return {'checkpoint':checkpoint,'commit':record,'commitSHA256':file_sha(checkpoint/'COMMIT.json')}

    def state(self):
        state=validate_state(read_json(self.root/'STATE.json'))
        current=self.inspect(state['latest'])
        if current['commit']['metadata'].get('runState')!=state:
            raise ValueError('Committed model and published run cursor differ')
        for name in {state['previous'],state['best']}-{None}: self.inspect(name)
        return state

    def publish(self,model,state,binding,save_checkpoint):
        validate_state(state)
        if (self.root/'STATE.json').exists():
            published=validate_state(read_json(self.root/'STATE.json'))
            if (published['sequence'],published['latest'])!=(state['sequence'],state['latest']):
                raise ValueError('Publication must extend the original current generation')
            # Finish any interrupted *old* retirement before making another
            # generation. Preserve a single expected pending successor.
            self.prune(published,allow_pending=True)
        if self.footprint()+SNAPSHOT_ALLOWANCE>STORE_LIMIT:
            raise ValueError('Owned rolling store would exceed512MiB; preserve residues for review')
        if shutil.disk_usage(self.root).free<15*1024**3+SNAPSHOT_ALLOWANCE:
            raise ValueError('Preserved15GiB reserve plus next complete snapshot required')
        name=f"g{state['sequence']+1:08d}"; p=self.generation(name)
        if not p.exists():
            p.mkdir(); create_json(p/'OWNED.json',{'schemaVersion':1,'generation':name,'armSHA256':self.binding})
        elif read_json(p/'OWNED.json')!={'schemaVersion':1,'generation':name,'armSHA256':self.binding}:
            raise ValueError('Existing pending snapshot is not this exact transaction')
        out={**state,'sequence':state['sequence']+1,'previous':state['latest'],'latest':name}
        meta={**binding,'epoch':out['epoch'],'cursor':out['cursor'],'updates':out['updates'],
              'validationPending':out['validationPending'],'testEvaluations':0,'runState':out}
        # Reuse of a fully committed pending generation is implemented by the
        # already tested tensor checkpoint writer: exact metadata+bitwise state.
        save_checkpoint(model,p,meta)
        self.preserve_partials(p,f"checkpoint-{out['updates']:09d}")
        checked=self.inspect(name)
        if checked['commit']['metadata']!=meta: raise ValueError('New committed state readback differs')
        atomic_json(self.root/'STATE.json',out)
        self.prune(out)
        return out

    def preserve_partials(self,generation,checkpoint):
        """Reversible relocation of this owned generation's interrupted temps.

        A fresh invocation may reuse an identical commit after a failed pointer
        promotion. Unique old temporary files must neither block that commit
        nor disappear later when its obsolete generation is retired.
        """
        for p in list(generation.iterdir()):
            if p.name in {'OWNED.json','LATEST.json',checkpoint}: continue
            plain(p)
            pointer=p.name.startswith('LATEST.partial-') and p.is_file()
            payload=bool(re.fullmatch(r'checkpoint-[0-9]{9}\.partial-.+',p.name)) and p.is_dir()
            if not(pointer or payload): raise ValueError('Unrecognized generation residue is preserved in place')
            files=[p] if pointer else list(p.iterdir())
            if payload and any(f.name not in {'tensors.pt','COMMIT.json'} for f in files):
                raise ValueError('Unrecognized partial payload members')
            manifest={f.name:{'size':regular(f).st_size,'sha256':file_sha(f)} for f in files}
            if sum(v['size'] for v in manifest.values())>128*1024**2: raise ValueError('Partial payload exceeds bound')
            folder=self.root/'ORPHANS'
            if not folder.exists(): folder.mkdir()
            plain(folder)
            target=Path(tempfile.mkdtemp(prefix=generation.name+'-',dir=folder))
            create_json(target/'INTENT.json',{'armSHA256':self.binding,'sourceGeneration':generation.name,'name':p.name,'files':manifest})
            os.rename(p,target/p.name)
            create_json(target/'MOVED.json',{'complete':True,'preserved':p.name})

    def pending(self):
        """Read an already completed successor after STATE promotion failed.

        Return only a content-verified candidate. The native caller must still
        load/validate its exact tensors and bindings before promote_pending().
        Incomplete uniquely named writer temps remain preserved for the next
        publication; no timeout or process guess establishes completion.
        """
        state=self.state();name=f"g{state['sequence']+1:08d}";p=self.generation(name)
        if not p.exists(): return None
        if read_json(p/'OWNED.json')!={'schemaVersion':1,'generation':name,'armSHA256':self.binding}:
            raise ValueError('Pending successor ownership differs')
        complete=[d for d in p.iterdir() if re.fullmatch(r'checkpoint-[0-9]{9}',d.name)]
        if not complete:return None
        if len(complete)!=1:raise ValueError('Only one committed pending model permitted')
        child=complete[0];plain(child)
        if {f.name for f in child.iterdir()}!={'COMMIT.json','tensors.pt'}:raise ValueError('Pending committed model incomplete')
        record=read_json(child/'COMMIT.json');meta=record['metadata'];candidate=validate_state(meta['runState'])
        if (record.get('complete')is not True or meta.get('sustainedArmSHA256')!=self.binding
            or candidate['sequence']!=state['sequence']+1 or candidate['latest']!=name or candidate['previous']!=state['latest']
            or not 0<=candidate['updates']-state['updates']<=6016 or candidate['validatedEpochs']-state['validatedEpochs']not in (0,1)
            or regular(child/'tensors.pt').st_size!=record['payloadBytes'] or file_sha(child/'tensors.pt')!=record['payloadSHA256']):
            raise ValueError('Pending successor state/content differs')
        return {'state':candidate,'checkpoint':child,'commitSHA256':file_sha(child/'COMMIT.json')}

    def promote_pending(self,candidate,verified_metadata):
        # Caller has just used the native weights_only loader with all expected
        # source/data/runtime/LR/RNG bindings and finite tensor checks.
        current=self.pending()
        if current!=candidate:raise ValueError('Pending successor changed during native verification')
        if read_json(candidate['checkpoint']/'COMMIT.json')['metadata']!=verified_metadata:
            raise ValueError('Native-verified pending metadata differs')
        p=candidate['checkpoint'].parent
        atomic_json(p/'LATEST.json',{'checkpoint':candidate['checkpoint'].name,'commitSHA256':candidate['commitSHA256']})
        self.preserve_partials(p,candidate['checkpoint'].name);self.inspect(p.name)
        atomic_json(self.root/'STATE.json',candidate['state']);self.prune(candidate['state'])
        return candidate['state']

    def prune(self,state,allow_pending=False):
        keep={state['latest'],state['previous'],state['best']}-{None}; removed=[]
        pending=self.root/'RETIRING.json'
        if pending.exists():
            intent=read_json(pending)
            name=intent.get('generation','')
            if not GENERATION.fullmatch(name) or int(name[1:])>=state['sequence'] or name in keep:
                raise ValueError('Pending retirement would remove a retained/future model')
            self.retire(intent); removed.append(intent)
        pending_names=set()
        for p in (self.root/'snapshots').iterdir():
            plain(p)
            if not p.is_dir() or not GENERATION.fullmatch(p.name): raise ValueError('Unknown owned-store entry; refuse deletion')
            if p.name in keep: continue
            if int(p.name[1:])>state['sequence']:
                if not allow_pending or p.name!=f"g{state['sequence']+1:08d}" or read_json(p/'OWNED.json')!={'schemaVersion':1,'generation':p.name,'armSHA256':self.binding}:
                    raise ValueError('Unexpected future snapshot is preserved, never retired')
                pending_names.add(p.name);continue
            # Full ownership, contents, digest and containment checks precede
            # recursive deletion; unknown or interrupted artifacts are retained.
            proof=self.inspect(p.name)
            if not p.resolve().is_relative_to((self.root/'snapshots').resolve()): raise ValueError('Deletion escapes owned snapshots')
            intent={'generation':p.name,'armSHA256':self.binding,
                    'files':{f.relative_to(p).as_posix():{'size':regular(f).st_size,'sha256':file_sha(f)} for f in p.rglob('*') if f.is_file()}}
            create_json(pending,intent)
            self.retire(intent); removed.append({'generation':p.name,'commitSHA256':proof['commitSHA256']})
        actual={p.name for p in (self.root/'snapshots').iterdir()}
        if actual!=keep|pending_names or len(keep)>3 or len(actual)>4 or self.footprint()>STORE_LIMIT:
            raise ValueError('Rolling retention readback differs')
        atomic_json(self.root/'RETENTION.json',{'kept':sorted(keep),'removed':removed,'bytes':self.footprint(),
                                             'boundBytes':STORE_LIMIT,'recoverablePredecessor':state['previous'],'best':state['best'],
                                             'pending':sorted(pending_names)})

    def retire(self,intent):
        """Recover only the exact already journaled old snapshot, file by file."""
        if intent.get('armSHA256')!=self.binding: raise ValueError('Retirement arm differs')
        p=self.generation(intent['generation']); files=intent['files']
        children={n.split('/')[0] for n in files if '/'in n}
        if len(files)!=4 or len(children)!=1:
            raise ValueError('Exact four owned checkpoint files required for retirement')
        child=next(iter(children))
        if not re.fullmatch(r'checkpoint-[0-9]{9}',child) or set(files)!={'OWNED.json','LATEST.json',child+'/COMMIT.json',child+'/tensors.pt'}:
            raise ValueError('Canonical retirement inventory required')
        if any(type(row.get('size'))is not int or not 0<=row['size']<=128*1024**2 or not re.fullmatch('[0-9a-f]{64}',row.get('sha256','')) for row in files.values()):
            raise ValueError('Bounded exact retirement sizes/digests required')
        if p.exists():
            plain(p)
            entries=list(p.rglob('*'))
            for f in entries: plain(f)
            if not {f.relative_to(p).as_posix() for f in entries if f.is_dir()}<={child}:
                raise ValueError('Unknown directory in interrupted retirement')
            actual={f.relative_to(p).as_posix() for f in p.rglob('*') if f.is_file()}
            if not actual<=set(files): raise ValueError('Unknown files in interrupted retirement')
            for name in actual:
                parts=Path(name).parts
                if Path(name).is_absolute() or '..'in parts or '\\'in name or ':'in name: raise ValueError('Retirement member escapes generation')
                f=p.joinpath(*name.split('/')); row=files[name]
                if regular(f).st_size!=row['size'] or file_sha(f)!=row['sha256']: raise ValueError('Interrupted retirement file differs')
            for name in sorted(actual): (p/name).unlink()
            for directory in sorted((d for d in p.rglob('*') if d.is_dir()),key=lambda d:len(d.parts),reverse=True):
                plain(directory); directory.rmdir()
            p.rmdir()
        append_json(self.root/'RETIREMENT.jsonl',{'completed':intent['generation'],'armSHA256':self.binding,'intentSHA256':value_sha(intent)})
        (self.root/'RETIRING.json').unlink()

    def validation(self,epoch,model_generation,evaluate):
        proof=self.inspect(model_generation); p=self.root/'VALIDATION.json'
        records=read_json(p) if p.exists() else {}
        header={'epoch':epoch,'generation':model_generation,'commitSHA256':proof['commitSHA256'],
                'armSHA256':self.binding,'examples':60000,'split':'original-undeformed-training'}
        reused=str(epoch)in records
        if reused:
            result=records[str(epoch)]
            if any(result.get(k)!=v for k,v in header.items()): raise ValueError('Existing validation belongs to different model')
        else:
            observation=evaluate(); result={**header,**observation}
            if any(result.get(k)!=v for k,v in header.items()) or type(result.get('errors'))is not int or not 0<=result['errors']<=60000:
                raise ValueError('Exact validation count/model header required')
            records[str(epoch)]=result
            if len(records)>800: raise ValueError('Finite800-epoch validation journal')
            atomic_json(p,records)
        return {**result,'reusedFromJournal':reused}

    def certify_complete(self,state):
        validate_state(state)
        if state['epoch']!=800 or state['updates']!=48000000 or state['cursor']!=0 or state['best']is None or state['validationPending']:
            raise ValueError('Exactly complete validated arm required')
        if self.state()!=state: raise ValueError('Final test needs the genuinely committed complete state')
        validations=read_json(self.root/'VALIDATION.json')
        if set(validations)!={str(i) for i in range(800)}:
            raise ValueError('All800 actual validation records required')
        for i in range(800):
            row=validations[str(i)]
            if (row.get('epoch')!=i or row.get('armSHA256')!=self.binding or row.get('examples')!=60000
                or row.get('split')!='original-undeformed-training' or type(row.get('errors'))is not int
                or not 0<=row['errors']<=60000 or not GENERATION.fullmatch(row.get('generation',''))
                or not re.fullmatch('[0-9a-f]{64}',row.get('commitSHA256',''))):
                raise ValueError('Actual validation journal header/count differs')
        best_epoch=min(range(800),key=lambda i:(validations[str(i)]['errors'],i))
        winner=validations[str(best_epoch)]
        if (state['bestEpoch'],state['bestErrors'],state['best'])!=(best_epoch,winner['errors'],winner['generation']):
            raise ValueError('Only predeclared earliest minimum can select final model')
        selected=self.inspect(state['best'])
        if selected['commitSHA256']!=winner['commitSHA256']:
            raise ValueError('Selected model differs from actual validation journal')
        return {'passed':True,'validatedEpochs':800,'trainingUpdates':48000000,
                'bestEpoch':best_epoch,'bestErrors':winner['errors'],'selectedGeneration':state['best'],
                'selectedCommitSHA256':selected['commitSHA256'],'validationSHA256':file_sha(self.root/'VALIDATION.json')}

    def reserve_test(self,state,approved):
        if approved is not True:raise ValueError('Separate explicit final-test approval required')
        selected=self.certify_complete(state)
        record={'state':'reserved','armSHA256':self.binding,'selectedGeneration':state['best'],
                'selectedCommitSHA256':selected['selectedCommitSHA256'],'selectedEpoch':state['bestEpoch'],
                'validationErrors':state['bestErrors'],'trainingUpdates':48000000,'testExamples':10000}
        # Exclusive reservation precedes even reading test bytes. An interrupted
        # reservation remains consumed; no reset or automatic second evaluation.
        create_json(self.root/'FINAL_TEST.json',record)
        return record

    def finish_test(self,reservation,observation):
        if read_json(self.root/'FINAL_TEST.json')!=reservation or reservation.get('state')!='reserved':
            raise ValueError('Original final-test reservation differs')
        if type(observation.get('errors'))is not int or not 0<=observation['errors']<=10000 or observation.get('examples')!=10000:
            raise ValueError('Exactly10000 genuine test predictions required')
        result={**reservation,'state':'complete','observation':observation,'testEvaluations':1}
        atomic_json(self.root/'FINAL_TEST.json',result)
        return result
