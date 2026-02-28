from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Literal
import torch

from .constants import DEFAULT_OUTPUT_ROOT, default_dino_checkpoint, default_dino_repo, default_sam_checkpoint


@dataclass
class ApexConfig:
    data_dir: str = ""
    local_db_path: str = ""
    output_root: str = DEFAULT_OUTPUT_ROOT
    max_cases: int = 3
    max_slices: int = 8
    test_labels: list[int] = field(default_factory=lambda: [1])
    retrieval_rank: int = 2
    retrieval_topk: int = 5
    retrieval_skip_self: bool = True
    force_input_size: int = 256
    bbox_size: int = 112
    prompt_mode: Literal["voronoi"] = "voronoi"
    enable_dino_freq_fusion: bool = True
    dino_gate_quantile: float = 0.9
    sam_checkpoint: str = field(default_factory=default_sam_checkpoint)
    dinov3_checkpoint: str = field(default_factory=default_dino_checkpoint)
    dinov3_repo: str = field(default_factory=default_dino_repo)
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    verbose_log: bool = False
    save_debug_viz: bool = False

    def __post_init__(self) -> None:
        self.dataset = "CHAOS_MR_T2"
        self.p1 = 1
        self.p99 = 99
        self.tau_in = 8
        self.tau_band = 6
        self.delta = 3
        self.sigmas = [1, 2, 4, 8]
        self.w_log = 0.6
        self.w_grad = 0.4
        self.roi_dino_region_enable = True
        self.roi_dino_region_min_cover = 0.020403061055619683
        self.roi_dino_region_min_candidates = 3
        self.roi_dino_region_dilate = 1
        self.roi_dino_region_close = 3
        self.roi_dino_region_erode = 1
        self.roi_dino_region_keep_peak_component = True
        self.dino_region_min_cover_ratio = 0.66
        self.dino_region_expand_iters = 24
        self.dino_region_ignore_support_area = True
        self.dino_hf_match_enable = True
        self.dino_hf_weight = 0.6
        self.premask_mode = "chamfer"
        self.premask_matched_only = False
        self.premask_as_output = False
        self.enable_dino_gate = True
        self.dino_gate_margin = 12
        self.edge_valid_thresh = 0.2
        self.n_angle_bins = 8
        self.rotations_deg = [-20, -10, 0, 10, 20]
        self.chamfer_stride = 4
        self.chamfer_min_valid_ratio = 0.6
        self.boundary_sample_step = 3
        self.fit_mode = "inscribed"
        self.r_fg = 8
        self.r_bg = 12
        self.r_mid = 6
        self.w_band = 20
        self.rw_beta = 90.0
        self.premask_min_cover_ratio = 0.66
        self.premask_expand_iters = 24
        self.neg_boundary_width = 1
        self.closed_refine_enable = True
        self.closed_edge_quantile = 0.8427100442938211
        self.closed_edge_dilate = 1
        self.closed_edge_close = 2
        self.closed_min_area_ratio = 0.002
        self.closed_pre_dilate = 6
        self.closed_hole_bonus = 0.25
        self.closed_hole_penalty = 0.25
        self.closed_max_mean_dist_frac = 0.25
        self.closed_area_min_ratio = 0.3766092133697905
        self.closed_area_max_ratio = 1.5874625902177333
        self.closed_dino_weight = 0.029911857534322267
        self.closed_dino_mask_weight = 0.35
        self.closed_dino_enable = True
        self.closed_dino_quantile = 0.9802925954031122
        self.closed_dino_dilate = 1
        self.closed_dino_close = 2
        self.closed_dino_expand_enable = False
        self.closed_dino_expand_gate_dilate = 24
        self.closed_dino_expand_target_ratio = 1.0
        self.closed_shape_weight = 0.35
        self.closed_dino_edge_enable = True
        self.closed_dino_edge_weight = 0.5
        self.closed_dino_edge_gamma = 2.0
        self.closed_quantile_step = 0.03
        self.closed_quantile_iters = 6
        self.scales = [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4]
        self.sim_num_regions = 24
        self.sim_top_regions = 16
        self.sim_scaler = 5.0
        self.M_pos_cand = 200
        self.M_neg_cand = 200
        self.K_pos = 6
        self.K_neg = 8
        self.kmeans_max_iter = 50
        self.prompt_branch = self.prompt_mode
        self.pos_mode = "fps_voronoi"
        self.min_dist = 8
        self.candidate_downsample = 2
        self.use_distance_transform = True
        self.d0 = 3
        self.enable_voronoi_partition = True
        self.voronoi_assign_method = "kd_tree"
        self.save_voronoi_viz = False
        self.voronoi_viz_dir = "debug/voronoi"
        self.voronoi_viz_alpha = 0.45
        self.voronoi_boundary_thickness = 1
        self.r = 5
        self.k = 2
        self.seed = 2021
        self.sam2_checkpoint = self.sam_checkpoint
        self.dinov3_model_name = "dinov3_vitl16"
        self.sam_input_size = self.force_input_size
        self.dino_size = 512
        self.dino_patch_size = 16
        self.dino_freq_level = 1
        self.dino_freq_mix_ratio = 0.35
        self.dino_freq_mix_mode = "support_high"
        self.enable_freq_style = False
        self.freq_style_level = 1
        self.freq_mix_ratio = 0.35
        self.freq_mix_mode = "support_high"
        self.save_freq_viz = False
        self.freq_style_affect_shape = False
        self.output_root = self.output_root
        self.viz_minimal_only = True
        self.viz_keep_stages = {"support_mask", "query_mask", "pre_mask", "points", "pred_mask", "pred_vs_gt_color"}

    @classmethod
    def from_cli_args(cls, args: Any) -> "ApexConfig":
        return cls(
            data_dir=args.data_dir,
            local_db_path=args.local_db_path,
            output_root=args.output_root,
            max_cases=args.max_cases,
            max_slices=args.max_slices,
            test_labels=list(args.test_labels),
            retrieval_rank=args.retrieval_rank,
            sam_checkpoint=args.sam_checkpoint,
            dinov3_checkpoint=args.dinov3_checkpoint,
            dinov3_repo=args.dinov3_repo,
            device=args.device,
        )

    def public_dict(self) -> dict[str, Any]:
        return asdict(self)
