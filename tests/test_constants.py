from sinergym.utils.constants import *


def test_normalization_dicts():
    # 5ZONE
    assert (all(norm_variable in RANGES_5ZONE.keys()
                for norm_variable in DEFAULT_5ZONE_OBSERVATION_VARIABLES))

    # DATACENTER
    assert (all(norm_variable in RANGES_DATACENTER.keys()
                for norm_variable in DEFAULT_DATACENTER_OBSERVATION_VARIABLES))

    # OFFICEMEDIUM
    assert (all(norm_variable in RANGES_OFFICE.keys()
                for norm_variable in DEFAULT_OFFICE_OBSERVATION_VARIABLES))

    # WAREHOUSE
    assert (all(norm_variable in RANGES_WAREHOUSE.keys()
                for norm_variable in DEFAULT_WAREHOUSE_OBSERVATION_VARIABLES))
