from diffusers import FluxFillPipeline, FluxTransformer2DModel
from MoeTesting.MoeTransformer.attn_processor import SkipAttnProcessor, MoEFluxAttnProcessor2_0
from MoeTesting.MoeTransformer.utils import init_transformer_adapter
from args_helper import parse_args

def main(args):
    # pipeline = FluxFillPipeline.from_pretrained(args.pretrained_model_name_or_path)
    transformer=FluxTransformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    
    init_transformer_adapter(transformer, cross_attn_cls=MoEFluxAttnProcessor2_0)

if __name__ == '__main__':
    args = parse_args()
    main(args)
