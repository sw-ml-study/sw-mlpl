Phase 5: Animation + transitions.

Current-step sculpture animates using D3.js transitions: matrix multiply (input matrices slide together, output grows from intersection), activation function (elements shift -- negative squash to zero for ReLU), reshape (elements rearrange into new shape). Past steps freeze in place as camera advances.

D3 drives per-element color/height changes on instanced meshes or texture maps within Three.js sculptures. Pages rebuild required.