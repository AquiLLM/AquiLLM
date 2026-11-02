import json
import torch
import traceback
from aquillm_vllm_h100.gdn.native import launch

with torch.inference_mode():
    state = torch.zeros((9,48,128,128),device='cuda')
    a_log = torch.nn.Parameter(torch.full((48,),-1.0,device='cuda'))
    dt = torch.nn.Parameter(torch.full((48,),-0.2,device='cuda',dtype=torch.float16))
    def value(shape):
        return torch.full(shape,0.1,device='cuda',dtype=torch.float16)
    kwargs = dict(A_log=a_log,dt_bias=dt,a=value((5,48)),b=value((5,48)),
        q=value((1,5,16,128)),k=value((1,5,16,128)),v=value((1,5,48,128)),
        initial_state=state,cu_seqlens=torch.tensor([0,5],device='cuda',dtype=torch.int32),
        ssm_state_indices=torch.tensor([[1,2,3,4,5]],device='cuda',dtype=torch.int32),
        num_accepted_tokens=torch.tensor([1],device='cuda',dtype=torch.int32))
    try:
        launch(**kwargs)
    except BufferError as error:
        print(json.dumps(dict(expected_failure=True,error=str(error),traceback=traceback.format_exc(),
            a_log_requires_grad=a_log.requires_grad,dt_bias_requires_grad=dt.requires_grad,
            state_unmodified=bool(torch.count_nonzero(state)==0)),indent=2))
    else:
        raise RuntimeError('Expected cold-compile Parameter export failure was not reproduced')
