"""Tiny real-filesystem journal/retention tests, no Torch/device or dataset."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parent))
import paper_runstate as r


def fake_checkpoint(model,root,metadata):
    """Only tensor serialization is replaced; actual RunStore is exercised."""
    child=root/f"checkpoint-{metadata['updates']:09d}"
    value={'schemaVersion':1,'complete':True,'metadata':metadata,'sizes':[1,1],
           'payloadBytes':len(model),'payloadSHA256':hashlib.sha256(model).hexdigest()}
    if child.exists():
        if r.read_json(child/'COMMIT.json')!=value: raise ValueError('Different pending committed state')
    else:
        child.mkdir(); (child/'tensors.pt').write_bytes(model); r.create_json(child/'COMMIT.json',value)
    r.atomic_json(root/'LATEST.json',{'checkpoint':child.name,'commitSHA256':r.file_sha(child/'COMMIT.json')})
    return child


def boundary(s,epoch):
    return {**s,'epoch':epoch,'cursor':0,'updates':epoch*60000,'validatedEpochs':epoch-1,
            'validationPending':True,'nextCheckpointAt':epoch*60000+6000}


def advanced(s):
    u=s['updates']+32
    return {**s,'updates':u,'cursor':s['cursor']+32,'nextCheckpointAt':(u//6000+1)*6000}


class Journal(unittest.TestCase):
    def make(self,d): return r.RunStore(Path(d)/'arm',{'armId':'fixture','fixedProtocol':True})

    def generations(self,store):
        s=store.publish(b'fixture-model',boundary(r.initial_state(0),1),{},fake_checkpoint)
        s=store.publish(b'fixture-model',r.after_validation(s,10),{},fake_checkpoint)
        s=store.publish(b'fixture-model',advanced(s),{},fake_checkpoint)
        return s

    def test_earliest_validation_only_selection(self):
        s=boundary(r.initial_state(0),1); s.update(latest='g00000001',sequence=1)
        s=r.after_validation(s,10); self.assertEqual((s['best'],s['bestEpoch']),('g00000001',0))
        s=boundary(s,2); s.update(latest='g00000002',sequence=2)
        tied=r.after_validation(s,10); self.assertEqual(tied['best'],'g00000001')
        better=r.after_validation(s,9); self.assertEqual((better['best'],better['bestEpoch']),('g00000002',1))
        with self.assertRaises(ValueError): r.after_validation(tied,1)

    def test_normalized_boundary_matches_original_checkpoint_contract(self):
        s=boundary(r.initial_state(0),800); s.update(validatedEpochs=799)
        r.validate_state(s); self.assertEqual(s['updates'],48000000)
        with self.assertRaises(ValueError): r.validate_state({**s,'cursor':60000})

    def test_three_generations_keep_best_previous_and_latest(self):
        with tempfile.TemporaryDirectory() as d:
            store=self.make(d); s=self.generations(store); old_previous=s['previous']
            s=store.publish(b'fixture-model',advanced(s),{},fake_checkpoint)
            actual={p.name for p in (store.root/'snapshots').iterdir()}
            self.assertEqual(actual,{s['latest'],s['previous'],s['best']}); self.assertEqual(len(actual),3)
            self.assertNotIn(old_previous,actual); self.assertEqual(store.state(),s)
            self.assertTrue((store.root/'RETIREMENT.jsonl').exists())

    def test_interrupted_pointer_reuses_exact_commit_and_preserves_partial(self):
        with tempfile.TemporaryDirectory() as d:
            store=self.make(d); s=r.initial_state(0)
            original=r.atomic_json
            def crash(path,value):
                if Path(path).name=='STATE.json': raise OSError('injected state pointer interruption')
                return original(path,value)
            with patch.object(r,'atomic_json',side_effect=crash):
                with self.assertRaises(OSError): store.publish(b'fixture-model',s,{},fake_checkpoint)
            generation=store.generation('g00000001'); (generation/'LATEST.partial-interrupted').write_bytes(b'preserve me')
            before=r.file_sha(next(generation.glob('checkpoint-*'))/'tensors.pt')
            s=store.publish(b'fixture-model',s,{},fake_checkpoint)
            self.assertEqual(before,r.file_sha(store.inspect(s['latest'])['checkpoint']/'tensors.pt'))
            preserved=list((store.root/'ORPHANS').glob('*/LATEST.partial-interrupted'))
            self.assertEqual(len(preserved),1); self.assertEqual(preserved[0].read_bytes(),b'preserve me')

    def test_interrupt_retirement_recovers_only_journaled_files(self):
        with tempfile.TemporaryDirectory() as d:
            store=self.make(d); s=self.generations(store); victim=store.generation(s['previous'])
            original=Path.unlink; failed=[]
            def crash(path,*a,**kw):
                if path.name=='tensors.pt' and path.is_relative_to(victim) and not failed:
                    failed.append(True); raise OSError('injected hard-stop location')
                return original(path,*a,**kw)
            with patch.object(Path,'unlink',new=crash):
                with self.assertRaises(OSError): store.publish(b'fixture-model',advanced(s),{},fake_checkpoint)
            self.assertTrue((store.root/'RETIRING.json').exists())
            current=store.state(); store.prune(current)
            self.assertFalse(victim.exists()); self.assertFalse((store.root/'RETIRING.json').exists())
            self.assertEqual({p.name for p in (store.root/'snapshots').iterdir()},{current['latest'],current['previous'],current['best']})

    def test_completed_pending_successor_needs_verified_metadata_before_promotion(self):
        with tempfile.TemporaryDirectory() as d:
            store=self.make(d);binding={'sustainedArmSHA256':store.binding}
            s=store.publish(b'fixture-model',r.initial_state(0),binding,fake_checkpoint)
            atomic=r.atomic_json
            def crash(path,value):
                if Path(path).name=='STATE.json':raise OSError('injected postcommit state interruption')
                return atomic(path,value)
            with patch.object(r,'atomic_json',side_effect=crash):
                with self.assertRaises(OSError):store.publish(b'fixture-model',advanced(s),binding,fake_checkpoint)
            self.assertEqual(store.state(),s);candidate=store.pending();self.assertEqual(candidate['state']['updates'],32)
            with self.assertRaises(ValueError):store.promote_pending(candidate,{'wrong':True})
            metadata=r.read_json(candidate['checkpoint']/'COMMIT.json')['metadata']
            restored=store.promote_pending(candidate,metadata)
            self.assertEqual(store.state(),restored);self.assertEqual(restored['sequence'],2)

    def test_unknown_content_and_budget_refuse_without_deletion(self):
        with tempfile.TemporaryDirectory() as d:
            store=self.make(d); s=self.generations(store); old=store.generation(s['previous']); (old/'unknown').write_bytes(b'keep')
            with self.assertRaises(ValueError): store.publish(b'fixture-model',advanced(s),{},fake_checkpoint)
            self.assertEqual((old/'unknown').read_bytes(),b'keep')
        with tempfile.TemporaryDirectory() as d:
            store=self.make(d)
            with patch.object(store,'footprint',return_value=r.STORE_LIMIT):
                with self.assertRaises(ValueError): store.publish(b'x',r.initial_state(0),{},fake_checkpoint)
            self.assertEqual(list((store.root/'snapshots').iterdir()),[])

    def test_validation_journal_reuse_and_model_binding(self):
        with tempfile.TemporaryDirectory() as d:
            store=self.make(d); s=store.publish(b'fixture-model',boundary(r.initial_state(0),1),{},fake_checkpoint)
            first=store.validation(0,s['latest'],lambda:{'errors':10,'seconds':.1})
            self.assertFalse(first['reusedFromJournal'])
            again=store.validation(0,s['latest'],lambda:(_ for _ in ()).throw(AssertionError('must not rescore')))
            self.assertTrue(again['reusedFromJournal']); self.assertEqual(again['errors'],10)
            with self.assertRaises(ValueError): store.validation(1,s['latest'],lambda:{'errors':0,'examples':10000})

    def test_final_test_reservation_is_one_use_and_requires_complete_arm(self):
        with tempfile.TemporaryDirectory() as d:
            store=self.make(d); s=self.generations(store)
            with self.assertRaises(ValueError): store.reserve_test(s,True)
            final={**s,'epoch':800,'cursor':0,'updates':48000000,'validatedEpochs':800,'validationPending':False,'nextCheckpointAt':48006000}
            with self.assertRaises(ValueError): store.reserve_test(final,False)
            final=store.publish(b'fixture-model',final,{},fake_checkpoint)
            with self.assertRaises((ValueError,FileNotFoundError)): store.reserve_test(final,True)
            proof=store.inspect(final['best'])
            records={str(i):{'epoch':i,'generation':final['best'],'commitSHA256':proof['commitSHA256'],
                            'armSHA256':store.binding,'examples':60000,'split':'original-undeformed-training','errors':10+i} for i in range(800)}
            r.create_json(store.root/'VALIDATION.json',records)
            reservation=store.reserve_test(final,True)
            with self.assertRaises(FileExistsError): store.reserve_test(final,True)
            result=store.finish_test(reservation,{'errors':35,'examples':10000,'seconds':.1})
            self.assertEqual(result['testEvaluations'],1)
            with self.assertRaises(ValueError): store.finish_test(reservation,{'errors':0,'examples':10000})


if __name__=='__main__': unittest.main(verbosity=2)
