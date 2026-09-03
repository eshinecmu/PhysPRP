import sys
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1]))
from main import run_stage


class MainInterfaceTest(TestCase):
    @patch("main.subprocess.run")
    def test_stage_only_formats_and_calls_existing_command(self, run):
        root = Path.cwd()
        run_stage("stage", {"argv": ["tool", "--input", "{rgb}"]}, {"rgb": "frame.png"}, root)
        run.assert_called_once_with(["tool", "--input", "frame.png"], cwd=root.resolve(), check=True)


if __name__ == "__main__":
    main()
