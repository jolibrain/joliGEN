def get_override_options_names(remaining_args):
    return_options_names = []

    for arg in remaining_args:
        if arg.startswith("--"):
            return_options_names.append(arg[2:])

    return return_options_names
