"""Project config file."""
import os
import utils


class Config:
    """Config class with global project variables."""

    def __init__(self, **kwargs):
        """Global config file for normalization experiments."""
        self.db_log_files = 'db_logs'

        # DB
        self.db_ssh_forward = False
        machine_name = os.uname()[1]
        if (machine_name != 'serrep7'):
            # Docker container or master p-node
            self.db_ssh_forward = True
        else:
            self.db_ssh_forward = False
        # self.db_ssh_forward = False

        # Create directories if they do not exist
        check_dirs = [
            self.db_log_files
        ]
        [utils.make_dir(x) for x in check_dirs]

    def __getitem__(self, name):
        """Get item from class."""
        return getattr(self, name)

    def __contains__(self, name):
        """Check if class contains field."""
        return hasattr(self, name)
