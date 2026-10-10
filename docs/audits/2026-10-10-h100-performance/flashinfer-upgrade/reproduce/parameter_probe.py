import contextlib
import json
import torch
from cutlass.cute.runtime import from_dlpack

rows = []
for parameter_dtype in (torch.float16,torch.float32):
    parameter = torch.nn.Parameter(torch.full((48,),-1.0,device='cuda',dtype=parameter_dtype))
    for inference in (False,True):
        with torch.inference_mode() if inference else contextlib.nullcontext():
            for detach in (False,True):
                value = parameter.detach() if detach else parameter
                row = dict(dtype=str(parameter_dtype),inference_mode=inference,detached=detach,
                           requires_grad=value.requires_grad,same_storage=value.data_ptr()==parameter.data_ptr())
                try:
                    tensor = from_dlpack(value,assumed_align=16)
                    row['exported'] = True
                except Exception as error:
                    row.update(exported=False,error_type=type(error).__name__,error=str(error))
                rows.append(row)
print(json.dumps(rows,indent=2))
