"""Deployment helpers for query-independent BEV attention values."""


RAW_BEV_INPUT_NAMES = [
    'lss_bev_feats',
    'radar_bev_feats',
]

PRECOMPUTED_BEV_INPUT_NAMES = [
    'lss_bev_value',
    'radar_bev_value',
]


def enable_precomputed_bev_values(decoder_layer):
    """Make a decoder layer consume projected BEV values directly."""
    for sampling in (
            decoder_layer.sampling_lss_bev,
            decoder_layer.sampling_radar_bev):
        sampling._deploy_value_preprojected = True
        sampling.attention._deploy_value_preprojected = True


def precompute_bev_values(decoder_layer, lss_bev_feats, radar_bev_feats):
    """Prepare the two values shared by all recurrent decoder iterations."""
    lss_bev_value = decoder_layer.sampling_lss_bev \
        .precompute_attention_value(lss_bev_feats)
    radar_bev_value = decoder_layer.sampling_radar_bev \
        .precompute_attention_value(radar_bev_feats)
    return lss_bev_value, radar_bev_value
