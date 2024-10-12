import openai

class ImageToText:
    def __init__(self, api_key, gpt_version="gpt-3.5-turbo"):
        self.template = self.initialize_template()
        openai.api_key = api_key
        self.gpt_version = gpt_version

    def initialize_template(self):
        prompt_prefix_1 = """Here we have 4-tuple [x1,y1,w,h] to depict the position of box that frames the objects or persons, where [x1,y1] means coordinate of the upper left corner of the box and [w,h] means the width and length of the box. The structure of people semantic is like "{{<symbol>:[P..] , <coordinate>:[x1,y1,w,h] , <caption>:[caption text of the people] , <age>:[age text of the people] , <gender>:[gender text of the people]}}". The structure of objects semantic is like "{{<symbol>:[O..] , <coordinate>:[x1,y1,w,h] , <caption>:[caption text of the object]}}". Generate only an informative and nature paragraph based on the given information (a,b,c,d) and following rules:\n"""
        prompt_suffix = """\n\nThere are some rules:\n- Pay more attention to the people semantic, which have reference <P>.\n- Depict the spatial relationships between individuals and objects, as well as the spatial relationships between people.\n- Must use symbols <O..> and <P..> when referring to objects and people.\n- Do not use coordinates [x1,y1,w,h], [x1,y1], [w,h] or numbers to show position information of each object. \n- Pay more attention to the social scene and describe the social event in detail. Explain how each person and object contributes to the social event.\n- No more than 15 sentences.\n- Only use one paragraph."""
        prompt_prefix_2 = """\n a. Image Resolution:  """
        prompt_prefix_3 = """\n b. Image Caption: """
        prompt_prefix_4 = """\n c. Image Scene: """
        prompt_prefix_5 = """\n d. People and Objects Semantic: """
        template = f"{prompt_prefix_1}{prompt_prefix_2}{{width}}X{{height}}{prompt_prefix_3}{{caption}}{prompt_prefix_4}{{image_caption_scene}}{prompt_prefix_5}{{region_semantic}}{prompt_suffix}"
        return template
    
    def paragraph_summary_with_gpt(self, caption, image_caption_scene, region_semantic, width, height):
        question = self.template.format(width=width, height=height, caption=caption, region_semantic=region_semantic,image_caption_scene=image_caption_scene)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print('\nStep3, Paragraph Summary with GPT-3:')
        print('\033[1;34m' + "Question:".ljust(10) + '\033[1;36m' + question + '\033[0m')
        completion = openai.ChatCompletion.create(
            model=self.gpt_version, 
            messages = [
            {"role": "system", "content": "You are an expert in generating only one naturally fluent and flawless paragraph based on a set of statements.\n\nYou must follow these rules:\n- illustrate the spatial relationship and depict the interaction between different people.\n- Do not use any coordinate to describe.\n- Must use symbols <O..> and <P..> when referring to objects and people."},
            {"role": "user", "content" : question}],
            temperature = 0,
            top_p = 1,
            max_tokens = 2048,
            frequency_penalty=0,
            presence_penalty=0
        )

        print('\033[1;34m' + "ChatGPT Response:".ljust(18) + '\033[1;32m' + completion['choices'][0]['message']['content'] + '\033[0m')
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return completion['choices'][0]['message']['content'], question

    def paragraph_summary_with_gpt_debug(self, caption, dense_caption, width, height):
        question = self.template.format(width=width, height=height, caption=caption, dense_caption=dense_caption)
        print("paragraph_summary_with_gpt_debug:")
        return question
