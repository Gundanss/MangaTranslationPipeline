from manga_pipeline.db import Database


def test_database_task_lifecycle(tmp_path):
    database = Database(tmp_path / "test.db")
    database.create_task(
        {
            "id": "task-1",
            "name": "测试",
            "config": {"source_language": "ja", "target_language": "zh-CN"},
            "output_dir": str(tmp_path / "output"),
        },
        [
            {
                "id": "image-1",
                "relative_path": "001.png",
                "input_path": str(tmp_path / "in.png"),
                "output_path": str(tmp_path / "out.png"),
                "context_path": str(tmp_path / "context.pkl"),
                "regions_path": str(tmp_path / "regions.json"),
            }
        ],
    )
    database.update_task("task-1", status="running", progress=0.5)
    database.update_image("image-1", status="completed", progress=1.0)
    database.log("task-1", "INFO", "ocr", "识别完成", "image-1", {"texts": ["原文"]})

    task = database.get_task("task-1")
    assert task["status"] == "running"
    assert task["images"][0]["status"] == "completed"
    assert database.get_logs("task-1")[0]["details"] == {"texts": ["原文"]}
