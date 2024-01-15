import argparse
from typing import TypeVar, Generic
from models import get_models_names, get_option_setter


def get_models_parsers():
    """
    Create a dict {model_name: parser} in which each parser hold arguments
    specific to a model
    """
    model_names = get_models_names()
    model_parsers = {}

    for model_name in model_names:
        parser = argparse.ArgumentParser()
        model_option_setter = get_option_setter(model_name)

        try:
            model_parsers[model_name] = model_option_setter(parser, True)
        except:
            pass

    return model_parsers


# Help formatters


class FilterArgumentParser(argparse.ArgumentParser):
    """Parser that accepts only the option related to given topic"""

    def __init__(self, keep_topics=None, remove_topics=None, **kwargs):
        super(FilterArgumentParser, self).__init__(**kwargs)
        self.keep_topics = keep_topics
        self.remove_topics = remove_topics

    def add_argument(self, *args, **kwargs):
        opt_name = args[0]
        opt_removed = False
        if opt_name.startswith("--"):
            # filter applies
            if self.remove_topics:
                for topic in self.remove_topics:
                    if opt_name.startswith("--" + topic + "_"):
                        opt_removed = True
                        break

            if self.keep_topics:
                opt_found = False
                for topic in self.keep_topics:
                    if opt_name.startswith("--" + topic + "_"):
                        opt_found = True
                        break
                if not opt_found:
                    opt_removed = True

        if not opt_removed:
            super().add_argument(*args, **kwargs)


class CustomHelpAction(argparse.Action):
    def get_class(options_class):
        class Tailored(CustomHelpAction):
            def __init__(
                self,
                option_strings,
                dest=argparse.SUPPRESS,
                default=argparse.SUPPRESS,
                help=None,
            ):
                super(Tailored, self).__init__(
                    option_strings=option_strings,
                    dest=dest,
                    default=default,
                    help=help,
                )
                self.options_class = options_class

        return Tailored

    def __init__(
        self,
        option_strings,
        dest=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
        help=None,
    ):
        super(CustomHelpAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs="?",
            help=help,
        )

    def __call__(self, parser, namespace, value, option_string=None):
        options = self.options_class()
        topic = value
        if not options.topic_exists(topic):
            print("Unknown topic: %s\n" % topic)
            topic = None

        parser = options.get_topic_parser(topic)
        parser.print_help()

        subtopics = options.get_topics(topic)
        if len(subtopics) > 0:
            print(
                "\n\nSelect topic to get help on associated options (--help [TOPIC]):\n"
            )
            BOLD = "\033[1m"
            END = "\033[0m"
            for key in subtopics:
                subtopic = subtopics[key]
                subtopic_desc = subtopic["title"] if "title" in subtopic else ""
                print("\t%s%s%s\t\t%s" % (BOLD, key, END, subtopic_desc))

        parser.exit()


def set_custom_help(parser, options_class):
    """
    Change --help into this app's custom help
    """
    parser.register("action", "help", CustomHelpAction.get_class(options_class))
    parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )
