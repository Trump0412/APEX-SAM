from apex_sam.cli.build_local_db import build_parser as build_db_parser
from apex_sam.cli.eval import build_parser as build_eval_parser


def test_eval_cli_help_builds():
    parser = build_eval_parser()
    assert parser.prog is not None


def test_build_db_cli_help_builds():
    parser = build_db_parser()
    assert parser.prog is not None
