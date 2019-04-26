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
