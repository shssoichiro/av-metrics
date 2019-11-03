//! From Netflix VMAF source repo: https://github.com/Netflix/vmaf

use itertools::Itertools;
use lazy_static::lazy_static;
use serde::Deserialize;
use serde_pickle::{from_slice, Value};
use std::collections::HashMap;

/// The string contents of the VMAF 'default' model that this library
/// was compiled with.
const VMAF_DEF_MODEL_STR: &str =
    include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/data/vmaf_v0.6.1.pkl"));

/// The string contents of the associated `.model` file for the `default` model.
const VMAF_DEF_SVM_STR: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/vmaf_v0.6.1.pkl.model"
));

lazy_static! {
    static ref VMAF_DEF_MODEL: VmafModel = parse_model(VMAF_DEF_MODEL_STR);
    static ref VMAF_4K_MODEL: VmafModel = parse_model(VMAF_4K_MODEL_STR);
    static ref VMAF_DEF_SVM: VmafSvmModel = parse_svm(VMAF_DEF_SVM_STR);
    static ref VMAF_4K_SVM: VmafSvmModel = parse_svm(VMAF_4K_SVM_STR);
}

/// The string contents of the VMAF '4K' model that this library
/// was compiled with.
const VMAF_4K_MODEL_STR: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/vmaf_4k_v0.6.1.pkl"
));

/// The string contents of the associated `.model` file for the `4K` model.
const VMAF_4K_SVM_STR: &str = include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/data/vmaf_4k_v0.6.1.pkl.model"
));

#[derive(Debug, Clone, Deserialize)]
struct VmafModelWrapper {
    model_dict: VmafModel,
}

#[derive(Debug, Clone, Deserialize)]
pub(crate) struct VmafModel {
    model_type: String,
    feature_names: Vec<String>,
    norm_type: NormType,
    slopes: Option<Vec<f64>>,
    intercepts: Option<Vec<f64>>,
    score_clip: Option<Vec<f64>>,
    score_transform: Option<HashMap<String, Value>>,
}

#[derive(Debug, Clone, Copy, Deserialize, PartialEq)]
#[serde(rename_all(deserialize = "snake_case"))]
pub(crate) enum NormType {
    None,
    LinearRescale,
}

pub(crate) fn get_vmaf_def_model() -> &'static VmafModel {
    &*VMAF_DEF_MODEL
}

pub(crate) fn get_vmaf_4k_model() -> &'static VmafModel {
    &*VMAF_4K_MODEL
}

fn parse_model(model: &str) -> VmafModel {
    let wrapper: VmafModelWrapper =
        from_slice(model.as_bytes()).expect("Failed to parse VMAF model");
    let model = wrapper.model_dict;
    assert_eq!(model.model_type, "LIBSVMNUSVR", "Unrecognized model type");
    if model.norm_type == NormType::LinearRescale {
        assert!(
            model.slopes.is_some(),
            "Slopes must be set for norm type linear rescale"
        );
        assert!(
            model.intercepts.is_some(),
            "Intercepts must be set for norm type linear rescale"
        );
    }
    model
}

#[derive(Debug, Clone)]
pub(crate) struct VmafSvmModel {
    svm_type: SvmType,
    kernel_type: KernelType,
    /// number of classes, = 2 in regression/one class svm
    nr_class: usize,
    /// total #SV
    total_sv: usize,
    /// constants in decision functions (`rho[nr_class*(nr_class-1)/2]`)
    rho: Vec<f64>,
    /// pairwise probability information
    prob_a: Vec<f64>,
    /// pairwise probability information
    prob_b: Vec<f64>,
    /// SVs (`SV[total_sv]`)
    svs: Vec<Sv>,
}

pub(crate) fn get_vmaf_def_svm() -> &'static VmafSvmModel {
    &*VMAF_DEF_SVM
}

pub(crate) fn get_vmaf_4k_svm() -> &'static VmafSvmModel {
    &*VMAF_4K_SVM
}

fn parse_svm(model: &str) -> VmafSvmModel {
    let headers: HashMap<&str, &str> = model
        .lines()
        .take_while(|&line| line != "SV")
        .map(|line| {
            let split = line.splitn(2, ' ').collect_vec();
            (split[0], split[1])
        })
        .collect();
    let svm_type = SvmType::parse(headers["svm_type"]);
    let kernel_type = KernelType::parse(headers["kernel_type"], &headers);
    let nr_class = headers["nr_class"]
        .parse()
        .expect("Failed to parse nr_class header");
    let total_sv = headers["total_sv"]
        .parse()
        .expect("Failed to parse nr_class header");
    let rho = headers
        .get("rho")
        .map(|rhos| {
            rhos.split(' ')
                .map(|rho| rho.parse::<f64>().unwrap())
                .collect_vec()
        })
        .unwrap_or_else(Vec::new);
    let prob_a = headers
        .get("probA")
        .map(|rhos| {
            rhos.split(' ')
                .map(|rho| rho.parse::<f64>().unwrap())
                .collect_vec()
        })
        .unwrap_or_else(Vec::new);
    let prob_b = headers
        .get("probB")
        .map(|rhos| {
            rhos.split(' ')
                .map(|rho| rho.parse::<f64>().unwrap())
                .collect_vec()
        })
        .unwrap_or_else(Vec::new);

    let svs = model
        .lines()
        .skip(headers.len() + 1)
        .map(|line| {
            let coeffs = line
                .split(' ')
                .take(nr_class - 1)
                .map(|coeff| coeff.parse::<f64>().unwrap())
                .collect_vec();
            let nodes = line
                .split(' ')
                .skip(nr_class - 1)
                .map(|nodes| {
                    let split = nodes.split(':').collect_vec();
                    SvNode {
                        index: split[0].parse().unwrap(),
                        value: split[1].parse().unwrap(),
                    }
                })
                .collect_vec();
            Sv { coeffs, nodes }
        })
        .collect();

    VmafSvmModel {
        svm_type,
        kernel_type,
        nr_class,
        total_sv,
        rho,
        prob_a,
        prob_b,
        svs,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum SvmType {
    CSvc,
    NuSvc,
    OneClass,
    EpsilonSvr,
    NuSvr,
}

impl SvmType {
    fn parse(ty: &str) -> Self {
        match ty {
            "c_svc" => SvmType::CSvc,
            "nu_svc" => SvmType::NuSvc,
            "one_class" => SvmType::OneClass,
            "epsilon_svr" => SvmType::EpsilonSvr,
            "nu_svr" => SvmType::NuSvr,
            _ => {
                panic!("Unsupported SVM type");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum KernelType {
    Linear,
    Polynomial { degree: i32, gamma: f64, coef0: f64 },
    Rbf { gamma: f64 },
    Sigmoid { gamma: f64, coef0: f64 },
    Precomputed,
}

impl KernelType {
    fn parse(ty: &str, headers: &HashMap<&str, &str>) -> Self {
        match ty {
            "linear" => KernelType::Linear,
            "polynomial" => KernelType::Polynomial {
                degree: headers["degree"].parse().unwrap(),
                gamma: headers["gamma"].parse().unwrap(),
                coef0: headers["coef0"].parse().unwrap(),
            },
            "rbf" => KernelType::Rbf {
                gamma: headers["gamma"].parse().unwrap(),
            },
            "sigmoid" => KernelType::Sigmoid {
                gamma: headers["gamma"].parse().unwrap(),
                coef0: headers["coef0"].parse().unwrap(),
            },
            "precomputed" => KernelType::Precomputed,
            _ => {
                panic!("Unsupported kernel type");
            }
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Sv {
    /// Number of coefficients = number of classes - 1
    coeffs: Vec<f64>,
    nodes: Vec<SvNode>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct SvNode {
    index: usize,
    value: f64,
}
