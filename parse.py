import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    # ── General ──────────────────────────────────────────────────────────
    parser.add_argument('--model_name', type=str, required=True,
                        help='ViTIC | AlphaRec | LightGCN | MF | MultVAE | SGL | XSimGCL')
    parser.add_argument('--dataset', type=str, required=True,
                        help='amazon_clothing | amazon_beauty')
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--saveID', type=str, default='run')
    parser.add_argument('--max_epoch', type=int, default=500)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--verbose', type=float, default=5)
    parser.add_argument('--test_only', action='store_true')
    parser.add_argument('--clear_checkpoints', action='store_true')
    parser.add_argument('--no_wandb', action='store_true')

    # ── Training ─────────────────────────────────────────────────────────
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--weight_decay', type=float, default=1e-6)
    parser.add_argument('--dropout', type=float, default=0.1)

    # ── Evaluation ───────────────────────────────────────────────────────
    parser.add_argument('--rs_type', type=str, default='General')
    parser.add_argument('--Ks', type=int, default=20)
    parser.add_argument('--data_path', type=str, default='./data/General/')
    parser.add_argument('--neg_sample', type=int, default=256)
    parser.add_argument('--infonce', type=int, default=1)
    parser.add_argument('--train_norm', action='store_true')
    parser.add_argument('--pred_norm', action='store_true')
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--regs', type=float, default=1e-5)
    parser.add_argument('--max2keep', type=int, default=1)
    parser.add_argument('--candidate', action='store_true')
    parser.add_argument('--nodrop', action='store_true')

    args, _ = parser.parse_known_args()

    # ── ViTIC-specific ────────────────────────────────────────────────────
    if 'ViTIC' in args.model_name:
        parser.add_argument('--tau', type=float, default=0.2)
        parser.add_argument('--lm_model', type=str, default='qwenvl_8b',
                            choices=['qwentext_8b', 'qwenvl_8b', 'qwenvl_8b_image',
                                     'qwentext_qwenvlimg_8b'])
        parser.add_argument('--model_version', type=str, default='mlp',
                            choices=['mlp', 'homo'])

    # ── AlphaRec-specific ─────────────────────────────────────────────────
    if args.model_name == 'AlphaRec':
        parser.add_argument('--tau', type=float, default=0.15)
        parser.add_argument('--lm_model', type=str, default='v3')
        parser.add_argument('--model_version', type=str, default='mlp',
                            choices=['mlp', 'homo'])

    # ── MultVAE-specific ──────────────────────────────────────────────────
    if args.model_name == 'MultVAE':
        parser.add_argument('--total_anneal_steps', type=int, default=200000)
        parser.add_argument('--anneal_cap', type=float, default=0.2)
        parser.add_argument('--p_dim0', type=int, default=200)
        parser.add_argument('--p_dim1', type=int, default=600)

    # ── SGL-specific ──────────────────────────────────────────────────────
    if args.model_name == 'SGL':
        parser.add_argument('--lambda_cl', type=float, default=0.2)
        parser.add_argument('--temp_cl', type=float, default=0.15)
        parser.add_argument('--droprate', type=float, default=0.1)

    # ── XSimGCL-specific ─────────────────────────────────────────────────
    if args.model_name == 'XSimGCL':
        parser.add_argument('--lambda_cl', type=float, default=0.1)
        parser.add_argument('--temp_cl', type=float, default=0.15)
        parser.add_argument('--layer_cl', type=int, default=1)
        parser.add_argument('--eps_XSimGCL', type=float, default=0.1)

    args_full, _ = parser.parse_known_args()
    special_args = sorted(set(vars(args_full).keys()) - set(vars(args).keys()))
    return args_full, special_args
