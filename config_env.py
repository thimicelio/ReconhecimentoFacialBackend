import os


def configurar_ambiente_paddle():
    os.environ["FLAGS_use_mkldnn"] = "0"
    os.environ["FLAGS_use_pir_api"] = "0"
    os.environ["FLAGS_enable_pir_in_executor"] = "0"
    os.environ["FLAGS_allocator_strategy"] = "auto_growth"
