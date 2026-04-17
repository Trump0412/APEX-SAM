from apex_sam.cli.build_expert_database import build_parser as build_module1_parser
from apex_sam.cli.eval import build_parser as build_eval_parser
from apex_sam.cli.inference import build_parser as build_inference_parser
from apex_sam.cli.preprocess_dataset import build_parser as build_preprocess_parser


def test_eval_cli_help_builds():
    parser = build_eval_parser()
    assert parser.prog is not None


def test_module1_cli_help_builds():
    parser = build_module1_parser()
    assert parser.prog is not None


def test_inference_cli_help_builds():
    parser = build_inference_parser()
    assert parser.prog is not None


def test_preprocess_cli_help_builds():
    parser = build_preprocess_parser()
    assert parser.prog is not None
