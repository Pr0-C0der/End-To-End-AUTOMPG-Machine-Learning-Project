import pytest

from utils import (
    remove_non_integer_values,
    load_data,
    check_non_integer_values,
    add_parameters,
)


def test_remove():
    """
    Tests the remove_non_integer_values function.

    This function tests the remove_non_integer_values function by loading the data, removing non-integer values, and checking that the number of columns is equal to the number of columns that contain only integer values.

    Returns:
        None.
    """

    data = load_data("data/data.txt")

    data = remove_non_integer_values(data)

    integer_columns_length = 0

    for column in data.columns:
        non_integer_values = check_non_integer_values(data, column)
        integer_columns_length += 1
        assert len(non_integer_values) == 0

    assert len(data.columns) == integer_columns_length


def test_add_parameters():
    """
    Tests the add_parameters function.

    This function tests the add_parameters function by loading the data, removing non-integer values, adding the new parameters, and checking that the new parameters are in the data frame.

    Returns:
        None.
    """

    data = load_data("data/data.txt")
    data = remove_non_integer_values(data)

    added_parameters = [
        "power_to_weight",
        "acceleration_efficiency",
        "displacement_per_cylinder",
        "weight_per_cylinder",
    ]

    data = add_parameters(data)

    for parameter in added_parameters:
        assert parameter in data.columns


if __name__ == "__main__":
    pytest.main()
