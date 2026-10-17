"""Install a separately trained visual residual without changing frozen V1 weights."""
import torch
from .calibrated_visual_residual import CalibratedVisualResidual,corrected_field
from .checkpointing import file_sha256


def install_calibrated_residual(policy,path,checkpoint):
    state=torch.load(path,map_location='cpu',weights_only=True)
    if state.get('kind')!='calibrated-visual-residual-v2' or state['config'].get('smoke'):
        raise ValueError('A fully trained calibrated residual is required')
    if state['base_sha']!=file_sha256(checkpoint):raise ValueError('Residual base checkpoint differs')
    head=CalibratedVisualResidual(**state['config'].get('model_kwargs',{})).to(policy.device);head.load_state_dict(state['model'],strict=True)
    head.requires_grad_(False).eval();policy.model.add_module('calibrated_residual',head)
    original=policy.model.forward
    def forward(**inputs):
        inputs=dict(inputs);control=inputs.pop('visual_control','real')
        if control not in ('real','no-belief','zero','shuffle'):raise ValueError('Unknown visual control')
        if control in ('zero','shuffle'):
            for feature_key,valid_key in (('roi_features','roi_valid'),('history_features','history_valid')):
                if inputs.get(feature_key) is None:continue
                values=inputs[feature_key].clone();valid=inputs[valid_key]
                if control=='zero':values.zero_()
                else:
                    for batch in range(len(values)):
                        for scale in range(values.shape[2]):
                            ids=valid[batch,:,scale].nonzero().flatten()
                            permutation=torch.randperm(len(ids),generator=torch.Generator().manual_seed(0)).to(ids.device)
                            values[batch,ids,scale]=values[batch,ids[permutation],scale]
                inputs[feature_key]=values
        if not inputs.get('use_history',False):
            for key in ('history_features','history_valid','history_quality'):
                inputs[key]=torch.zeros_like(inputs[key])
        base=original(**inputs,visual_control='no-belief')
        if control=='no-belief':delta=torch.zeros_like(base.delta)
        else:
            keys=('roi_features','roi_valid','roi_quality','history_features','history_valid','history_quality','native_text','appearance_text','pose_world','bounds')
            public={key:inputs[key] for key in keys}
            delta,confidence,known=head(base,public,inputs['static_inputs'])
        base.delta=delta;base.belief=corrected_field(base.static,delta,inputs['static_inputs']['grid_xy'])
        return base
    policy.model.forward=forward
    previous=policy.predict
    def predict(state,instruction):
        result=previous(state,instruction)
        result.diagnostics.update(residual_model='calibrated-visual-residual-v2',residual_bound=head.bound,residual_centered=head.centered)
        return result
    policy.predict=predict
