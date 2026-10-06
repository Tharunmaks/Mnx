import argparse
import sys


def parse_args(argv):
    parser = argparse.ArgumentParser(description="Greet someone.")
    parser.add_argument("name", help="who to greet")
    parser.add_argument("--shout", action="store_true", help="use upper case")
    return parser.parse_args(argv)


def greet(name, shout=False):
    """Return a greeting for name."""
    message = f"Hello, {name}!"
    return message.upper() if shout else message


def main(argv=None):
    args = parse_args(argv if argv is not None else sys.argv[1:])
    print(greet(args.name, args.shout))
    return 0


if __name__ == "__main__":
    sys.exit(main())
