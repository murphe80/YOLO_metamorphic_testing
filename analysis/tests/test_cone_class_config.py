import tempfile
import unittest
from pathlib import Path

from utils.cone_class_config import DEFAULT_CONFIG_PATH, load_cone_class_config
from utils.cone_colour_class_transformation import read_ground_truth_labels_txt


class ConeClassConfigTests(unittest.TestCase):
    def test_dataset_specific_blue_ids_are_resolved_semantically(self):
        test_config = load_cone_class_config(DEFAULT_CONFIG_PATH, "test")
        competition_config = load_cone_class_config(
            DEFAULT_CONFIG_PATH, "comp_labelled"
        )

        self.assertEqual(test_config["class_name_to_id"]["blue_cone"], 3)
        self.assertEqual(competition_config["class_name_to_id"]["blue_cone"], 1)
        self.assertEqual(
            competition_config["class_id_to_name"][3], "large_orange_cone"
        )
        self.assertEqual(
            competition_config["transformations"]["blue_to_yellow"],
            {"blue_cone": "yellow_cone"},
        )

    def test_txt_reader_uses_selected_dataset_schema(self):
        competition_config = load_cone_class_config(
            DEFAULT_CONFIG_PATH, "comp_labelled"
        )
        with tempfile.TemporaryDirectory() as directory:
            label_path = Path(directory) / "frame.txt"
            label_path.write_text(
                "1 0.5 0.5 0.2 0.2\n"
                "3 0.25 0.25 0.1 0.1\n"
            )
            detections = read_ground_truth_labels_txt(
                str(label_path),
                (100, 200),
                competition_config["class_id_to_name"],
            )

        self.assertEqual(
            [detection["class_name"] for detection in detections],
            ["blue_cone", "large_orange_cone"],
        )

    def test_unknown_dataset_fails_closed(self):
        with self.assertRaisesRegex(ValueError, "no cone-class schema"):
            load_cone_class_config(DEFAULT_CONFIG_PATH, "unconfigured_dataset")


if __name__ == "__main__":
    unittest.main()
