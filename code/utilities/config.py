imitaition_conf = {
    'num_res_blocks' : 19,
    'pgn_dir' : './Lichess Elite Database',
    'time_limit' : 0.5,
    'batch_size' : 128,
    'epochs' : 2,
    'actor_lr' : 1e-5,
    'critic_lr' : 1e-5,
    'landa' : 0.8,
    'save_pth' : 'artifacts/weights.torch',
    'game_buffer_size' : int(5e4),
    'loss_report_rate' : 10
}

download_url = "https://storage.cse-sbu.ir/s/aETtMXL28RBPiL2/download/Lichess%20Elite%20Database.7z"
