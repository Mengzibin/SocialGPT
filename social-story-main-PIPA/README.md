# social-story-main-PIPA

## Dataset

### Overview
Download the PIPA dataset online, which includes the image set along with the corresponding relationship files: `relation_trainidx.txt`, `relation_train.txt`

## Code

### Key Variables
Configuration and changes are primarily made in the `main.py` file:

| Variable          | Description                                                 |
| ----------------- | ----------------------------------------------------------- |
| `image_folder`    | Location of the image folder after downloading PIPA dataset |
| `image_txt`       | Corresponding relationship file from PIPA dataset           |
| `dir_result`      | Folder location where the generated data will be stored     |
| `api_key`         | API key for OpenAI                                          |
| `category_number` | Number of relationship categories, 16 for PIPA dataset      |

## Run the Code

### Environment Setup
Create a new virtual environment using conda.

### Dependencies
Install all libraries listed in the `requirements.txt` file.

### Execution
Run the script using the command:
```bash
python main.py
```

## Result

### Generated Directory Structure
The generated folders are structured as follows, with `dir_result` being the top-level folder specified:

```
dir_result/
├── caption/                      # Stores text files generated during the process
│   ├── caption_answer/           # Stores results of relationship recognition accuracy
│   ├── caption_main/             # Stores initial results of object and person segmentation, generated stories, prompts, and relationships
│   ├── caption_relation/         # Stores relationship recognition results and dataset annotated ground truth
│   ├── caption_result_category/  # Stores statistics on relationship types identified, with columns for relationship type, correct and incorrect identifications, and missed relationships
│   └── caption_sam/              # Stores generated stories
├── original_example/             # Stores original images used for generating stories
├── story_example/                # Stores segmented item and person bounding box images from stories
├── story_example_mask/           # Stores segmented item and person mask images from stories
├── story_example_mask_people/    # Stores segmented person mask images from stories
└── story_example_people/         # Stores segmented person bounding box images from stories
```
