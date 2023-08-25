from sinergym.utils.constants import *


def test_normalization_dicts():
    # 5ZONE
    assert (all(norm_variable in RANGES_5ZONE.keys() for norm_variable in list(
        DEFAULT_5ZONE_VARIABLES.keys()) + list(DEFAULT_5ZONE_METERS.keys())))

    # DATACENTER
    assert (all(norm_variable in RANGES_DATACENTER.keys() for norm_variable in list(
        DEFAULT_DATACENTER_VARIABLES.keys()) + list(DEFAULT_DATACENTER_METERS.keys())))

    # OFFICEMEDIUM
    assert (all(norm_variable in RANGES_OFFICE.keys() for norm_variable in list(
        DEFAULT_OFFICE_VARIABLES.keys()) + list(DEFAULT_OFFICE_METERS.keys())))

    # WAREHOUSE
    assert (all(norm_variable in RANGES_WAREHOUSE.keys() for norm_variable in list(
        DEFAULT_WAREHOUSE_VARIABLES.keys()) + list(DEFAULT_WAREHOUSE_METERS.keys())))
