imitaition_conf = {
    'num_res_blocks' : 19,
    'pgn_dir' : './Lichess Elite Database',
    'time_limit' : 0.1,
    'batch_size' : 64,
    'epochs' : 1,
    'actor_lr' : 1e-5,
    'critic_lr' : 1e-5,
    'landa' : 0.8,
    'save_pth' : 'artifacts/weights.torch',
    'game_buffer_size' : int(1e4),
    'loss_report_rate' : 10
}

download_url = "https://storage.cse-sbu.ir/s/aETtMXL28RBPiL2"
