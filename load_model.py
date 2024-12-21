import CounTR.models_mae_cross as models_mae_cross
model = models_mae_cross.__dict__["mae_vit_base_patch16"](norm_pix_loss=False)

print(model)