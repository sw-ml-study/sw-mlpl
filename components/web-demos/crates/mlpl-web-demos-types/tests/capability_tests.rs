//! Phase 1 (local-gpu-agentic): demo capability tiers. Unlisted
//! demos default to CPU/live (runnable on the public demo); only
//! connect/GPU demos carry an override, with MLX and CUDA kept as
//! distinct devices so the UI groups + gates them separately.

use mlpl_web_demos_types::{Capability, Device, capability_for, demo_disabled};

#[test]
fn gating_keys_off_peer_device_set() {
    let cuda = capability_for("CUDA LoRA fine-tune");
    let mlx = capability_for("MLX LoRA fine-tune");
    let live = Capability::CPU_LIVE;
    // Live demos are always runnable; connect demos need a connection.
    assert!(!demo_disabled(&live, false, &[]));
    assert!(demo_disabled(&cuda, false, &[]));
    // Connected to a CUDA peer: the CUDA demo runs, the MLX demo stays
    // disabled (the peer does not offer mlx) -- gating off the real set.
    let cuda_peer = [Device::Cpu, Device::Cuda];
    assert!(!demo_disabled(&cuda, true, &cuda_peer));
    assert!(demo_disabled(&mlx, true, &cuda_peer));
    // Connected to an MLX peer: the reverse.
    let mlx_peer = [Device::Cpu, Device::Mlx];
    assert!(demo_disabled(&cuda, true, &mlx_peer));
    assert!(!demo_disabled(&mlx, true, &mlx_peer));
}

#[test]
fn unlisted_demo_defaults_to_cpu_live() {
    let c = capability_for("Totally Unlisted Demo");
    assert_eq!(c, Capability::CPU_LIVE);
    assert!(!c.requires_connect, "cpu/live demos run without a server");
    assert_eq!(c.device, Device::Cpu);
}

#[test]
fn contextual_ask_requires_connect_but_is_cpu() {
    let c = capability_for("Ask Ollama (contextual)");
    assert!(c.requires_connect, ":ask needs a connected mlpl-serve");
    assert_eq!(c.device, Device::Cpu, "Ollama call is device-agnostic");
}

#[test]
fn mlx_demo_is_mlx_device_and_connect_only() {
    let c = capability_for("MLX LoRA fine-tune");
    assert!(c.requires_connect);
    assert_eq!(c.device, Device::Mlx);
}

#[test]
fn mlx_and_cuda_are_distinct_devices() {
    assert_ne!(Device::Mlx, Device::Cuda);
    assert_ne!(Device::Mlx, Device::Cpu);
}
