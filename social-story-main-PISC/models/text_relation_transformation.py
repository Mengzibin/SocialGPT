import openai

str_to_label = {'friends':0,'family-members':1,'couple':2,'professional':3,'commercial':4,'no-relationship':5}

label_to_str = {0:'friends',1:'family-members',2:'couple',3:'professional',4:'commercial',5:'no-relationship'}

class TextRelationTransformationPIPA:
    def __init__(self, api_key, gpt_version="gpt-3.5-turbo"):
        # Load your big model here
        openai.api_key = api_key
        self.gpt_version = gpt_version
        self.template = self.initialize_template()

    def initialize_template(self):
        self.system = "You are an expert assistant in recognizing social relationships between people based on textual descriptions. \n\nYou must follow these rules:\n- The answer can only be one of the 6 listed social relationships.\n- Give the most likely answer and don't refuse to answer.\n- If can't decide, then randomly select one.\n- Output the final answer in the first sentence."
        prompt_prefix_1 = 'As a social relations expert, you have the skill to accurately identify the category of social relationships portrayed in an image based on its text description. Your expertise covers 6 distinct types of social relationships, with each pair of individuals falling under one of these 6 categories. Using the provided information, you draw inferences to determine the most likely type of social relationship depicted in an image. Your final output should be one of 6 distinct types of social relationships, defined as follows: {{<friends>, <family-members>, <couple>, <professional>, <commercial>, <no-relationship>}}.\n\nHere is the definition: \n'
        prompt_prefix_2 = '1. <friends>: A relationship between individuals who share common interests, care for each other, and engage in social activities together.\n'
        prompt_prefix_3 = '2. <family-members>: A relationship defined by blood or legal ties, such as marriage or adoption, where individuals share a household or familial bond.\n'
        prompt_prefix_4 = '3. <couple>: A romantic or committed partnership between two individuals, often characterized by love, trust, and a shared life plan.\n'
        prompt_prefix_5 = '4. <professional>: A relationship based on professional interaction, work, or collaboration, where individuals work together to achieve common goals or further their careers.\n'
        prompt_prefix_6 = '5. <commercial>: A relationship based on professional interaction, work, or collaboration, where individuals work together to achieve common goals or further their careers.\n'
        prompt_prefix_7 = '6. <no-relationship>: A lack of established connection or interaction between individuals or entities, often implying no shared interests, obligations, or engagements.\n'
        prompt_prefix_8 = '\n****************************************************************\n'
        prompt_prefix_9 = 'Example: \n[1. image description]:\nIn the image with a resolution of 640X480, a captivating scene unfolds as two women, <P1> and <P2>, elegantly dressed, engage in the culinary art of cooking a turkey in an oven. The women, positioned side by side, are focused on their task, their attention solely on the delicious bird roasting inside the oven. <P1>, a young adult woman, skillfully tends to the turkey, ensuring it cooks to perfection. Meanwhile, <P2>, a young person, also dressed in a dress, assists in the cooking process. The scene is further enriched by the presence of various objects that contribute to the ambiance. A man, <O1>, stands in front of a wall, adding a sense of depth to the image. A white cabinet, <O2>, adorned with a bird, stands nearby, providing a decorative touch. The centerpiece of the scene, the large turkey, <O3>, with its prominent head and legs, takes center stage, tantalizing the senses. A black plastic cover with a white stripe, <O4>, protects the turkey, ensuring its succulence. A white cabinet with two doors and a mirror, <O5>, adds a touch of elegance to the surroundings. A person, <O6>, stands in front of a bathroom mirror, perhaps preparing for the festivities. A black and white image of a black triangle, <O7>, adds an artistic element to the composition. Another man, <O8>, stands in front of a wall adorned with pictures, creating a visually appealing backdrop. A white wall with pictures and a white refrigerator, <O9>, adds a sense of domesticity to the scene. Finally, a red and black statue sitting on top of a black background, <O10>, adds a touch of intrigue to the overall ambiance. Together, these individuals and objects create a captivating social event, where the art of cooking and the beauty of the surroundings converge to create a memorable experience.\n\n'
        prompt_prefix_10 = '[2. Question]: \nWhat are the most likely social relationships between P1 and P2? Choose only one from {{<friends>, <family-members>, <couple>, <professional>, <commercial>, <no-relationship>}}.\n\n'
        prompt_prefix_11 = '[3. Answer]: [The final answer is <friends>]. The description presents a scenario where two women are engaged in a shared activity of cooking a turkey. They seem to be working together in a domestic setting, possibly preparing for a social event. The image doesn’t provide direct indicators of familial ties or a professional or commercial relationship. There’s no evidence suggesting a romantic relationship either. Their collaboration and shared activity in a personal setting like a home lean more towards a friendship. It\'s also possible they could be family, but the description leans slightly more towards a friendly interaction, especially in the absence of explicit familial indicators.\n'
        prompt_prefix_12 = '****************************************************************\n\n'
        prompt_prefix_13 = '[1. image description]:\n'
        prompt_prefix_14 = '\n\n[2. Question]: \nWhat are the most likely social relationships between '
        prompt_prefix_15 = ' and '
        prompt_prefix_16 = '? Choose only one from {{<friends>, <family-members>, <couple>, <professional>, <commercial>, <no-relationship>}}.\n\n[3. Answer]:'
        template = f"{prompt_prefix_1}{prompt_prefix_2}{prompt_prefix_3}{prompt_prefix_4}{prompt_prefix_5}{prompt_prefix_6}{prompt_prefix_7}{prompt_prefix_8}{prompt_prefix_9}{prompt_prefix_10}{prompt_prefix_11}{prompt_prefix_12}{prompt_prefix_13}{{story}}{prompt_prefix_14}{{P1}}{prompt_prefix_15}{{P2}}{prompt_prefix_16}"
        return template

    def text_to_relation_gpt_model(self,story,P1,P2):
        question = self.template.format(story=story,P1=P1,P2=P2)
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        print('\nStep3, Paragraph Summary with GPT-3.5:')
        print('\033[1;34m' + "Question:".ljust(10) + '\033[1;36m' + question + '\033[0m')
        completion = openai.ChatCompletion.create(
            model=self.gpt_version,
            messages = [
            {"role": "system", "content": self.system},
            {"role": "user", "content" : question}],
            temperature = 0
        )

        print('\033[1;34m' + "ChatGPT Response:".ljust(18) + '\033[1;32m' + completion['choices'][0]['message']['content'] + '\033[0m')
        print('\033[1;35m' + '*' * 100 + '\033[0m')
        return completion['choices'][0]['message']['content'],question

