# SocialGPT

## Overview
SocialGPT leverages the large language model ChatGPT to extract stories and identify corresponding social relationships from image datasets. This involves working with two different datasets, PIPA and PISC, and applying long prompt optimization techniques to improve the effectiveness of prompt in  relationship identification.

### Projects
1. **social-story-main-PIPA**: Uses ChatGPT to extract stories and identify social relationships from the PIPA dataset.
2. **social-story-main-PISC**: Uses ChatGPT to extract stories and identify social relationships from the PISC dataset.
3. **GSPO-PIPA**: Applies Long Prompt Optimization with GSPO in the PIPA dataset.
4. **GSPO-PISC**: Applies Long Prompt Optimization with GSPO in the PISC dataset.

## Running the Code
Each component of the SocialGPT project is contained within its own sub-directory. Navigate to the appropriate directory and refer to its README.md for detailed instructions on how to run the code:

| Component                              | Description                                                  | Directory                 |
| -------------------------------------- | ------------------------------------------------------------ | ------------------------- |
| PIPA Story and Relationship Extraction | Uses ChatGPT for story extraction and relationship identification in PIPA. | `social-story-main-PIPA/` |
| PISC Story and Relationship Extraction | Uses ChatGPT for story extraction and relationship identification in PISC. | `social-story-main-PISC/` |
| GSPO for PIPA                          | Applies GSPO techniques for long prompt optimization in PIPA. | `GSPO-PIPA/`              |
| GSPO for PISC                          | Applies GSPO techniques for long prompt optimization in PISC. | `GSPO-PISC/`              |

Please follow the individual README.md files within each sub-directory for specific setup and execution instructions.