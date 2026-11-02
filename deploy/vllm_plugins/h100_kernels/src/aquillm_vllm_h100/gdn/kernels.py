"""Per-call scratch and GPU-only metadata for the narrow N=1 MTP bridge."""
import torch
import triton
import triton.language as tl


@triton.jit
def _metadata(Offsets, Indices, Accepted, Read, Meta, T:tl.constexpr,
              COLUMNS:tl.constexpr, SLOTS:tl.constexpr):
    start = tl.load(Offsets).to(tl.int64)
    end = tl.load(Offsets+1).to(tl.int64)
    accepted = tl.load(Accepted).to(tl.int64)
    valid = (start>=0)&(end>=start)&(end<=T)&(accepted>0)&(accepted<=COLUMNS)
    slot = tl.load(Indices+accepted-1, mask=valid, other=-1).to(tl.int64)
    active = valid & (end>start) & (slot>0) & (slot<SLOTS)
    tl.store(Read, tl.where(active,slot,-1))
    tl.store(Meta,start)
    tl.store(Meta+1,tl.where(active,end-start,0))


@triton.jit
def _pack_qkv(Q,K,V,PQ,PK,PV,Meta,T:tl.constexpr,
              QS0:tl.constexpr,QS1:tl.constexpr,QS2:tl.constexpr,
              KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,
              VS0:tl.constexpr,VS1:tl.constexpr,VS2:tl.constexpr,BLOCK:tl.constexpr):
    x = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    t = x//(48*128)
    h = x//128%48
    d = x%128
    start = tl.load(Meta)
    length = tl.load(Meta+1)
    active = (t<T)&(t<length)
    qi = (start+t)*QS0+h*QS1+d*QS2
    ki = (start+t)*KS0+h*KS1+d*KS2
    vi = (start+t)*VS0+h*VS1+d*VS2
    q = tl.load(Q+qi,mask=active&(h<16),other=0).to(tl.bfloat16)
    k = tl.load(K+ki,mask=active&(h<16),other=0).to(tl.bfloat16)
    v = tl.load(V+vi,mask=active,other=0).to(tl.bfloat16)
    tl.store(PQ+(t*16+h)*128+d,q,mask=(t<T)&(h<16))
    tl.store(PK+(t*16+h)*128+d,k,mask=(t<T)&(h<16))
    tl.store(PV+x,v,mask=t<T)


@triton.jit
def _pack_gates(A,B,PA,PB,Meta,T:tl.constexpr,AS0:tl.constexpr,AS1:tl.constexpr,
                BS0:tl.constexpr,BS1:tl.constexpr,BLOCK:tl.constexpr):
    x = tl.arange(0,BLOCK)
    t = x//48
    h = x%48
    start = tl.load(Meta)
    length = tl.load(Meta+1)
    active = (t<T)&(t<length)
    a = tl.load(A+(start+t)*AS0+h*AS1,mask=active,other=0).to(tl.bfloat16)
    b = tl.load(B+(start+t)*BS0+h*BS1,mask=active,other=0).to(tl.bfloat16)
    tl.store(PA+x,a,mask=t<T)
    tl.store(PB+x,b,mask=t<T)


@triton.jit
def _scatter(Checkpoints,Pool,Indices,Read,Meta,T:tl.constexpr,SLOTS:tl.constexpr,
             SLOT_STRIDE:tl.constexpr,BLOCK:tl.constexpr):
    x = (tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)).to(tl.int64)
    length = tl.load(Meta+1)
    read = tl.load(Read)
    # A program owns one pool tile for all timesteps. Repeated destinations
    # are overwritten in increasing order, never raced by separate programs.
    for t in range(T):
        dst = tl.load(Indices+t).to(tl.int64)
        active = (read>0)&(t<length)&(dst>0)&(dst<SLOTS)&(x<48*128*128)
        value = tl.load(Checkpoints+t*(48*128*128)+x,mask=active,other=0)
        tl.store(Pool+dst*SLOT_STRIDE+x,value,mask=active)


@triton.jit
def _unpack(Packed,Output,Read,Meta,T:tl.constexpr,BLOCK:tl.constexpr):
    x = tl.program_id(0)*BLOCK + tl.arange(0,BLOCK)
    t = x//(48*128)
    start = tl.load(Meta)
    length = tl.load(Meta+1)
    read = tl.load(Read)
    active = (read>0)&(t<T)&(t<length)
    value = tl.load(Packed+x,mask=active,other=0)
    tl.store(Output+start*(48*128)+x,value.to(tl.float16),mask=active)


def prepare(q,k,v,a,b,initial_state,cu_seqlens,ssm_state_indices,num_accepted_tokens,**unused):
    t = q.shape[1]
    # Allocations belong to this invocation/captured graph. No mutable cache.
    packed = [torch.empty(shape,device=q.device,dtype=torch.bfloat16)
              for shape in (q.shape,k.shape,v.shape,(1,t,48),(1,t,48))]
    read = torch.empty((1,),device=q.device,dtype=torch.int64)
    metadata = torch.empty((2,),device=q.device,dtype=torch.int64)
    checkpoints = torch.empty((1,t,48,128,128),device=q.device,dtype=torch.float32)
    packed_out = torch.empty_like(packed[2])
    output = torch.empty_like(v, memory_format=torch.contiguous_format)
    _metadata[(1,)](cu_seqlens,ssm_state_indices,num_accepted_tokens,read,metadata,
                    t,ssm_state_indices.shape[1],initial_state.shape[0])
    _pack_qkv[(triton.cdiv(t*48*128,256),)](
        q,k,v,*packed[:3],metadata,t,*q.stride()[1:],*k.stride()[1:],*v.stride()[1:],256)
    _pack_gates[(1,)](a,b,*packed[3:],metadata,t,*a.stride()[-2:],*b.stride()[-2:],
                      triton.next_power_of_2(t*48))
    return packed,read,metadata,checkpoints,packed_out,output


def scatter_and_unpack(checkpoints,packed_out,output,pool,indices,read,metadata):
    t = checkpoints.shape[1]
    _scatter[(triton.cdiv(48*128*128,1024),)](
        checkpoints,pool,indices,read,metadata,t,pool.shape[0],pool.stride(0),1024)
    _unpack[(triton.cdiv(t*48*128,256),)](packed_out,output,read,metadata,t,256)
