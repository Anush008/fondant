import sys

import pandas as pd
from src.main import CreateNodes


def test_transform():
    """Test chunk component method."""
    input_dataframe = pd.DataFrame(
        {
            ("text", "documents"): [
                '{"id_": "1", "embedding": null, "metadata": {"source": "http://arxiv.org/pdf/1102.0183"}, "excluded_embed_metadata_keys": [], "excluded_llm_metadata_keys": [], "relationships": {}, "hash": "1a2Z", "text": "Lorem ipsum dolor sit amet, consectetuer adipiscing elit. Aenean commodo", "start_char_idx": null, "end_char_idx": null, "text_template": "{metadata_str}\\n\\n{content}", "metadata_template": "{key}: {value}", "metadata_seperator": "\\n", "class_name": "Document"}',
                '{"id_": "2", "embedding": null, "metadata": {"source": "http://arxiv.org/pdf/1102.0183"}, "excluded_embed_metadata_keys": [], "excluded_llm_metadata_keys": [], "relationships": {}, "hash": "3Re3", "text": "ligula eget dolor. Aenean massa. Cum sociis natoque penatibus et magnis dis", "start_char_idx": null, "end_char_idx": null, "text_template": "{metadata_str}\\n\\n{content}", "metadata_template": "{key}: {value}", "metadata_seperator": "\\n", "class_name": "Document"}',
                '{"id_": "3", "embedding": null, "metadata": {"source": "http://arxiv.org/pdf/1102.0183"}, "excluded_embed_metadata_keys": [], "excluded_llm_metadata_keys": [], "relationships": {}, "hash": "54t4", "text": "parturient montes, nascetur ridiculus mus. Donec quam felis, ultricies nec,", "start_char_idx": null, "end_char_idx": null, "text_template": "{metadata_str}\\n\\n{content}", "metadata_template": "{key}: {value}", "metadata_seperator": "\\n", "class_name": "Document"}'
                ],
        },
        index=pd.Index(["a", "b", "c"], name="id"),
    )

    component = CreateNodes(
        chunk_size=50,
        chunk_overlap=20
    )

    output_dataframe = component.transform(input_dataframe)

    return output_dataframe

    # expected_output_dataframe = pd.DataFrame(
    #     {
    #         ("text", "original_document_id"): ["a", "a", "b", "b", "c", "c"],
    #         ("text", "chunk"): [
    #             "Lorem ipsum dolor sit amet, consectetuer",
    #             "amet, consectetuer adipiscing elit. Aenean",
    #             "elit. Aenean commodo",
    #             "ligula eget dolor. Aenean massa. Cum sociis",
    #             "massa. Cum sociis natoque penatibus et magnis dis",
    #             "parturient montes, nascetur ridiculus mus. Donec",
    #             "mus. Donec quam felis, ultricies nec,",
    #         ],
    #     },
    #     index=pd.Index(["a_0", "a_1", "a_2", "b_0", "b_1", "c_0", "c_1"], name="id"),
    # )

    # pd.testing.assert_frame_equal(
    #     left=output_dataframe,
    #     right=expected_output_dataframe,
    #     check_dtype=False,
    # )