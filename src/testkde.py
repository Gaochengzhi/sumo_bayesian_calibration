from process_data import kl_divergence


def test_kl_divergence():
    #  "../output/data_raw/merge1/_cache.pkl",
    #     "../output/data_cache/merge_cache.pkl",
    a = kl_divergence(
        # "../output/data_cache/merge_cache.pkl", 
        "../output/data_raw/merge/_cache.pkl",
        "../output/data_raw/merge/_cache.pkl"
    )
    print(a)

test_kl_divergence()