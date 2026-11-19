import inspect
from pathlib import Path
import torch
import pytest
from transformers import MistralConfig, MistralForCausalLM

from prs_med15.models.global_tokens import GlobalTokenPooler, global_token_count
from prs_med15.models.query_encoder import QueryEncoder
from prs_med15.models.detail_retriever import DetailRetriever
from prs_med15.models.llm_bridge import (DeltaTokenEmbedding, SlotSpan, build_inputs_embeds,
    gather_seg_states, generate_with_seg_states)
from prs_med15.models.region_memory import EntityEntry, EntityRegistry, MaskedRegionPool
from prs_med15.training.losses import dialogue_mask_loss
from prs_med15.models.encoder import TinyConvEncoder
from prs_med15.models.slice_adapter import SliceAdapter
from prs_med15.models.mask_decoder import TinyMaskDecoder
from prs_med15.models.region_memory import RegionTokenizer
from prs_med15.models.model import PRSMed15
from prs_med15.data.dataset import DialogueDataset, DatasetConfig
from prs_med15.data.collate import collate_dialogues
from prs_med15.data.schema import Dialogue, Scan, ObjectRecord, Turn, Action
from prs_med15.training.trainer import save_checkpoint, load_checkpoint, cosine_scheduler
from prs_med15.cli import main as cli_main
from tests.fake_tokenizer import FakeTokenizer


def tiny_lm(vocab=100):
    torch.manual_seed(1)
    return MistralForCausalLM(MistralConfig(hidden_size=32,num_hidden_layers=1,
        num_attention_heads=4,num_key_value_heads=2,intermediate_size=64,vocab_size=vocab,
        max_position_embeddings=128,pad_token_id=0,eos_token_id=2,bos_token_id=1)).eval()


def test_global_budget_masked_pool_and_empty_slice():
    pool=GlobalTokenPooler(4,8,budget=32)
    x=torch.ones(1,5,8,8,4); valid=torch.ones(1,5,8,8,dtype=torch.bool); valid[:,4]=False
    out=pool(x,valid)
    assert out.tokens.shape==(1,32,8) and out.token_mask.sum()==16
    assert out.policy["depth_pair_rounds"]==2 and global_token_count(5,32)==32


def test_retriever_padding_invariance_and_query_api():
    lm=tiny_lm(); qe=QueryEncoder(lm.get_input_embeddings(),32,16,slots=3,layers=1,heads=4).eval()
    assert "answer" not in str(inspect.signature(qe.forward)) and "mask" not in str(inspect.signature(qe.forward)).replace("question_mask","").replace("region_mask","")
    q=qe(torch.tensor([[3,4]])); retr=DetailRetriever(4,16,32,layers=1,heads=4).eval()
    x=torch.randn(1,1,2,2,4); valid=torch.tensor([[[[1,1],[1,0]]]],dtype=torch.bool)
    a=retr(q,x,valid); x[0,0,1,1]=10000
    assert torch.allclose(a,retr(q,x,valid),atol=1e-6)


def test_placeholder_splice_and_delta_roundtrip(tmp_path: Path):
    lm=tiny_lm(); base=lm.get_input_embeddings(); delta=DeltaTokenEmbedding(base,[100,101])
    ids=torch.tensor([[4,100,100,5]]); labels=ids.clone(); values=torch.randn(2,32)
    sp=build_inputs_embeds(ids,[SlotSpan(0,[1,2],"g")],{"g":values},delta,labels=labels)
    assert torch.equal(sp.inputs_embeds[0,1:3],values) and (sp.labels[0,1:3]==-100).all()
    loss=delta(ids).sum(); loss.backward(); assert base.weight.grad is None and delta.delta.grad.abs().sum()>0
    state=delta.delta_state_dict(); other=DeltaTokenEmbedding(lm.get_input_embeddings(),[100,101]); other.load_delta_state_dict(state)
    assert torch.equal(delta(ids),other(ids))


def test_region_fallback_versioning_and_mask_loss_zero():
    pool=MaskedRegionPool(4); feat=torch.randn(2,3,3,4); vec,conf=pool(feat,torch.zeros(2,6,6))
    assert conf==0 and torch.equal(vec,pool.fallback)
    reg=EntityRegistry(); e=EntityEntry("S1","E1",1,"T1",torch.zeros(2),torch.zeros(1),torch.zeros(4),{},0.,"oracle")
    reg.create(e)
    with pytest.raises(ValueError): reg.update("S1","E1",EntityEntry("S1","E2",2,"T2",torch.zeros(2),torch.zeros(1),torch.zeros(4),{},0.,"predicted"))
    ref=torch.randn(2,requires_grad=True); loss=dialogue_mask_loss([],ref); assert loss.item()==0 and torch.isfinite(loss)


def test_seg_teacher_forced_matches_incremental_and_generate():
    lm=tiny_lm(); ids=torch.tensor([[1,7,8,9]])
    full=lm(ids,output_hidden_states=True,use_cache=False).hidden_states[-1][0,2]
    past=None; state=None
    for i in range(3):
        out=lm(ids[:,i:i+1],past_key_values=past,use_cache=True,output_hidden_states=True)
        past=out.past_key_values; state=out.hidden_states[-1][0,-1]
    assert torch.allclose(full,state,atol=1e-5)
    seq,states,pos=generate_with_seg_states(lm,8,input_ids=ids[:,:2],max_new_tokens=2,do_sample=False)
    # Greedy need not emit SEG; when it does, the gathered state is its own token state.
    if pos:
        manual=lm(seq,output_hidden_states=True).hidden_states[-1]
        assert torch.allclose(states,gather_seg_states(manual,pos),atol=1e-5)


@pytest.mark.parametrize("depth",[1,4])
def test_full_tiny_two_turn_model_backward(depth):
    tok=FakeTokenizer(); a1="<CREATE> <SCAN_REF> S1 <ENTITY> E1 [SEG] </CREATE>"
    a2="<CREATE> <SCAN_REF> S1 <ENTITY> E2 <REF> S1:E1 [SEG] </CREATE>"
    dlg=Dialogue("d","p","c","s","train",[Scan("S1","x",None,"CT","volume")],
      {"E1":ObjectRecord("o1","a"),"E2":ObjectRecord("o2","b")},[
      Turn("T1","segment a",a1,None,None,"S1",[Action("create","S1","E1","o1")]),
      Turn("T2","segment inside it",a2,None,None,"S1",[Action("create","S1","E2","o2",refs=["S1:E1"])],["E1"])])
    loader=lambda _:(torch.rand(3,depth,16,16),torch.ones(depth,16,16,dtype=torch.bool),{"unknown_geometry":True})
    mask=lambda _d,_e:torch.ones(depth,16,16,dtype=torch.bool)
    item=DialogueDataset([dlg],tok,DatasetConfig(scan_loader=loader,mask_loader=mask,d_max=4,
        global_budget=16,detail_tokens=4,region_tokens_per_entity=4))[0]
    lm=MistralForCausalLM(MistralConfig(hidden_size=32,num_hidden_layers=1,num_attention_heads=4,
        num_key_value_heads=2,intermediate_size=64,vocab_size=len(tok.vocab),max_position_embeddings=128))
    enc=TinyConvEncoder(feature_dim=8); adapter=SliceAdapter(8,heads=2,layers=1)
    model=PRSMed15(enc,adapter,GlobalTokenPooler(8,32,budget=16),
        QueryEncoder(lm.get_input_embeddings(),32,16,slots=4,layers=1,heads=4),
        DetailRetriever(8,16,32,layers=1,heads=4),lm,TinyMaskDecoder(8,32),
        MaskedRegionPool(8),RegionTokenizer(8,32,32),
        {name:tok.vocab[value] for name,value in (("global","{GLOBAL}"),("detail","{DETAIL}"),("region","{REGION}"))})
    out=model(collate_dialogues([item])); assert len(out.mask_logits)==2 and torch.isfinite(out.logits).all()
    loss=out.logits.float().mean()+sum(x.mean() for x in out.mask_logits); loss.backward()
    assert enc.net[0].weight.grad is not None


def test_checkpoint_roundtrip(tmp_path):
    model=torch.nn.Linear(3,2); opt=torch.optim.AdamW(model.parameters()); sched=cosine_scheduler(opt,1,5)
    before={k:v.clone() for k,v in model.state_dict().items()}; path=tmp_path/"state.pt"
    save_checkpoint(path,model,opt,sched,7,{"tag":"x"})
    with torch.no_grad(): model.weight.add_(10)
    step,extra=load_checkpoint(path,model,opt,sched)
    assert step==7 and extra["tag"]=="x" and all(torch.equal(model.state_dict()[k],v) for k,v in before.items())


def test_cli_heavy_guard(tmp_path,monkeypatch):
    config=tmp_path/"config.yaml"; config.write_text("data:\n  manifest: missing.jsonl\n")
    monkeypatch.delenv("PRSMED15_ALLOW_HEAVY",raising=False)
    with pytest.raises(SystemExit): cli_main(["train","--config",str(config)])
    assert cli_main(["prepare","--config",str(config),"--limit","1"])==0


def test_train_slab_visibility_exact_slots_and_no_duplicate_memory():
    tok=FakeTokenizer(); create=lambda e:f"<CREATE> <SCAN_REF> S1 <ENTITY> {e} [SEG] </CREATE>"
    dlg=Dialogue("slab-dialogue","p","c","s","train",[Scan("S1","x",None,"CT","volume")],
      {"E1":ObjectRecord("o1","a"),"E2":ObjectRecord("o2","b")},[
      Turn("T1","first",create("E1"),None,None,"S1",[Action("create","S1","E1","o1")]),
      Turn("T2","second",create("E2"),None,None,"S1",[Action("create","S1","E2","o2")],["E1"])])
    masks={"E1":torch.nn.functional.one_hot(torch.tensor(0),6)[:,None,None].expand(6,8,8).bool(),
           "E2":torch.nn.functional.one_hot(torch.tensor(5),6)[:,None,None].expand(6,8,8).bool()}
    loader=lambda _:(torch.rand(3,6,8,8),torch.ones(6,8,8,dtype=torch.bool),{})
    item=DialogueDataset([dlg],tok,DatasetConfig(scan_loader=loader,mask_loader=lambda _d,e:masks[e],
        d_max=3,global_budget=16,detail_tokens=4,region_tokens_per_entity=4,seed=2))[0]
    assert item["scans"][0].shape[1]==3 and item["geometry"][0]["slab_policy"]=="train_gt_visibility"
    assert sum(action["target"] is not None for action in item["actions"])==1
    # Two post-action blocks only: the in-window E1 block is not duplicated before turn 2.
    assert item["slot_positions"]["region"].numel()==8
    assert item["slot_positions"]["global"].numel()==global_token_count(3,16)
    assert item["slot_positions"]["detail"].numel()==8


def test_mistral_config_keeps_llava_rope_theta():
    from prs_med15.models.llm_bridge import mistral_config_from_llava
    raw = {"hidden_size": 64, "num_hidden_layers": 1, "num_attention_heads": 4,
           "num_key_value_heads": 2, "intermediate_size": 128, "vocab_size": 100,
           "rope_theta": 1000000.0, "model_type": "llava_mistral", "mm_hidden_size": 1024}
    config = mistral_config_from_llava(raw)
    theta = (getattr(config, "rope_parameters", None) or {}).get("rope_theta", getattr(config, "rope_theta", None))
    assert theta == 1000000.0
