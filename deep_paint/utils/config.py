"""Create and read config file for the device."""
import os
import logging
import configparser

import fire


def create_config():
    """Create config file for the device."""
    # Make deep_paint directory in user's home directory
    config_dir = os.path.join(os.path.expanduser('~'), '.deep_paint')
    try:
        os.mkdir(config_dir)
    except FileExistsError:
        pass
    config_file = os.path.join(config_dir, "config")
    if not os.path.exists(config_file):
        open(config_file, "w", encoding="utf-8").close()
    else:
        logging.error(f"Config file already exists at {config_file}")


def add_section(section_key: str, value: str):
    """Add key-value pair to config file."""
    # Read and validate config file
    config_file = os.path.join(os.path.expanduser('~'), ".deep_paint", "config")
    _validate_config(config_file)
    config = configparser.ConfigParser()
    config.read(config_file)

    # Add section and key-value pair
    section, key = section_key.split('.')
    if not config.has_section(section):
        config.add_section(section)
    config.set(section, key, value)
    if section == "paths":
        _create_path(value)

    # Write to config file
    with open(config_file, "w", encoding="utf-8") as f:
        config.write(f)


def get_value(key: str, section: str = "paths"):
    """Get value from config file."""
    # Read and validate config file
    config_file = os.path.join(os.path.expanduser('~'), ".deep_paint", "config")
    _validate_config(config_file)
    config = configparser.ConfigParser()
    config.read(config_file)
    return config.get(section, key)


def _create_path(path: str):
    if not os.path.exists(path):
        logging.info(f"Creating path {path}")
        os.makedirs(path)


def _validate_config(config_path):
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config file not found at {config_path}. Please create a config file first.")


if __name__ == '__main__':
    fire.Fire(create_config)
