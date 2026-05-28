use mlpl_core::{ActivationKind, LossKind, ValueTag};

#[test]
fn unit_variants_construct() {
    let _ = ValueTag::Logit;
    let _ = ValueTag::Probability;
    let _ = ValueTag::LogProbability;
    let _ = ValueTag::LearningRate;
    let _ = ValueTag::AttentionMap;
}

#[test]
fn loss_carries_kind() {
    let t = ValueTag::Loss {
        kind: LossKind::CrossEntropy,
    };
    match t {
        ValueTag::Loss { kind } => assert_eq!(kind, LossKind::CrossEntropy),
        _ => panic!("expected Loss variant"),
    }
}

#[test]
fn gradient_carries_wrt() {
    let t = ValueTag::Gradient {
        wrt: "W1".to_string(),
    };
    match &t {
        ValueTag::Gradient { wrt } => assert_eq!(wrt, "W1"),
        _ => panic!("expected Gradient variant"),
    }
}

#[test]
fn weight_carries_layer_and_name() {
    let t = ValueTag::Weight {
        layer: "encoder.linear_1".to_string(),
        name: "W".to_string(),
    };
    match &t {
        ValueTag::Weight { layer, name } => {
            assert_eq!(layer, "encoder.linear_1");
            assert_eq!(name, "W");
        }
        _ => panic!("expected Weight variant"),
    }
}

#[test]
fn bias_carries_layer() {
    let t = ValueTag::Bias {
        layer: "encoder.linear_1".to_string(),
    };
    match &t {
        ValueTag::Bias { layer } => assert_eq!(layer, "encoder.linear_1"),
        _ => panic!("expected Bias variant"),
    }
}

#[test]
fn activation_carries_layer_and_kind() {
    let t = ValueTag::Activation {
        layer: "encoder.tanh_0".to_string(),
        kind: ActivationKind::Tanh,
    };
    match &t {
        ValueTag::Activation { layer, kind } => {
            assert_eq!(layer, "encoder.tanh_0");
            assert_eq!(*kind, ActivationKind::Tanh);
        }
        _ => panic!("expected Activation variant"),
    }
}

#[test]
fn labels_carries_num_classes() {
    let t = ValueTag::Labels { num_classes: 10 };
    match t {
        ValueTag::Labels { num_classes } => assert_eq!(num_classes, 10),
        _ => panic!("expected Labels variant"),
    }
}

#[test]
fn display_name_stable_for_unit_variants() {
    assert_eq!(ValueTag::Logit.display_name(), "Logit");
    assert_eq!(ValueTag::Probability.display_name(), "Probability");
    assert_eq!(ValueTag::LogProbability.display_name(), "LogProbability");
    assert_eq!(ValueTag::LearningRate.display_name(), "LearningRate");
    assert_eq!(ValueTag::AttentionMap.display_name(), "AttentionMap");
}

#[test]
fn display_name_stable_for_struct_variants() {
    let g = ValueTag::Gradient {
        wrt: "W1".to_string(),
    };
    assert_eq!(g.display_name(), "Gradient");

    let w = ValueTag::Weight {
        layer: "l".to_string(),
        name: "W".to_string(),
    };
    assert_eq!(w.display_name(), "Weight");

    let b = ValueTag::Bias {
        layer: "l".to_string(),
    };
    assert_eq!(b.display_name(), "Bias");

    let a = ValueTag::Activation {
        layer: "l".to_string(),
        kind: ActivationKind::Relu,
    };
    assert_eq!(a.display_name(), "Activation");

    let l = ValueTag::Labels { num_classes: 3 };
    assert_eq!(l.display_name(), "Labels");

    let loss = ValueTag::Loss {
        kind: LossKind::Mse,
    };
    assert_eq!(loss.display_name(), "Loss");
}

#[test]
fn loss_kind_variants_present() {
    let _ = LossKind::CrossEntropy;
    let _ = LossKind::Mse;
    let _ = LossKind::KlDivergence;
    let _ = LossKind::Custom;
}

#[test]
fn activation_kind_variants_present() {
    let _ = ActivationKind::Tanh;
    let _ = ActivationKind::Relu;
    let _ = ActivationKind::Sigmoid;
    let _ = ActivationKind::Softmax;
}

#[test]
fn clone_preserves_data() {
    let t = ValueTag::Weight {
        layer: "l".to_string(),
        name: "W".to_string(),
    };
    let c = t.clone();
    assert_eq!(t, c);
}

#[test]
fn equality_distinguishes_metadata() {
    let a = ValueTag::Gradient {
        wrt: "W1".to_string(),
    };
    let b = ValueTag::Gradient {
        wrt: "W2".to_string(),
    };
    assert_ne!(a, b);
}

#[test]
fn serde_round_trip_unit_variant() {
    let t = ValueTag::Logit;
    let json = serde_json::to_string(&t).expect("serialize");
    let back: ValueTag = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(t, back);
}

#[test]
fn serde_round_trip_struct_variant() {
    let t = ValueTag::Weight {
        layer: "encoder.linear_1".to_string(),
        name: "W".to_string(),
    };
    let json = serde_json::to_string(&t).expect("serialize");
    let back: ValueTag = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(t, back);
}

#[test]
fn serde_round_trip_loss_with_kind() {
    let t = ValueTag::Loss {
        kind: LossKind::KlDivergence,
    };
    let json = serde_json::to_string(&t).expect("serialize");
    let back: ValueTag = serde_json::from_str(&json).expect("deserialize");
    assert_eq!(t, back);
}
