
def get_model_params(model):
    model_total_params = sum(p.numel() for p in model.parameters())
    model_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return model_total_params, model_total_trainable_params

