import inspect
import sys

from collections import ChainMap


def prepare_input_parameters(caller, default_parameters, custom_parameters=None):
    """Prepares an input parameter dictionary for operator benchmarks. Performs the union of default_parameters and
    custom_parameters i.e., takes the parameters provided, if any, in custom_parameters and replaces them in the
    default_parameters.

    Throws ValueError if custom_parameters contains a key not found in default_parameters.

    :param default_parameters: Dictionary of default parameters - name/value
    :param custom_parameters: (Optional) Dictionary of custom parameters - name/value. That will replace the corresponding
    parameter name/value in default_parameters.
    :param caller: str, Name of the caller (Operator Name) to be used in error message.
    :return: Dictionary of parameters which is a union of default_parameters and custom_parameters.
    """
    if custom_parameters is None:
        return default_parameters

    for key, value in custom_parameters.items():
        if key not in default_parameters.keys():
            raise ValueError("Invalid parameter provided for benchmarking operator - '{}'. "
                             "Given - '{}'. Supported - '{}'".format(caller, key, default_parameters.keys()))

        default_parameters[key] = value

    return default_parameters


def merge_map_list(map_list):
    """Merge all the Map in map_list into one final Map.

    Useful when you have a list of benchmark result maps and you want to prepare one final map combining all results.

    :param map_list: List of maps to be merged.
    :return: map where all individual maps in the into map_list are merged

    """
    return dict(ChainMap(*map_list))


def get_class_members_in_module(module_name):
    """Get list of class members in the given module. It returns only instantiable module names i.e.,
    non-abstract class names.

    :param module_name: Name of the module from where to list all the class member names.
    :return: tuple, (class_name, class_object)
    """
    members = inspect.getmembers(sys.modules[module_name],
                                 lambda member: inspect.isclass(member) and member.__module__ == module_name
                                                and not inspect.isabstract(member))
    return members
