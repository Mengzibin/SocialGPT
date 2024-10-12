import openai

str_to_label = {'father-child':0,'mother-child':1,'grandpa-grandchild':2,'grandma-grandchild':3,
                'friends':4,'siblings':5,'classmates':6,'lovers/spouses':7,
                'presenter-audience':8,'teacher-student':9,'trainer-trainee':10,'leader-subordinate':11,
                'band members':12,'dance team members':13,'sport team members':14,'colleagues':15}

label_to_str = {0:'father-child',1:'mother-child',2:'grandpa-grandchild',3:'grandma-grandchild',
                4:'friends',5:'siblings',6:'classmates',7:'lovers/spouses',
                8:'presenter-audience',9:'teacher-student',10:'trainer-trainee',11:'leader-subordinate',
                12:'band members',13:'dance team members',14:'sport team members',15:'colleagues'}

class TextRelationTransformationPIPA:
    def __init__(self, api_key, gpt_version="gpt-3.5-turbo"):
        # Load your big model here
        openai.api_key = api_key
        self.gpt_version = gpt_version
        self.template = self.initialize_template()

    def initialize_template(self):
        self.system = "You are an expert assistant in recognizing social relationships between people based on textual descriptions. \n\nYou must follow these rules:\n- The answer can only be one of the 16 listed social relationships.\n- Give the most likely answer and don't refuse to answer.\n- If can't decide, then randomly select one.\n- Output the final answer in the first sentence."
        prompt_prefix_1 = 'As a social relations expert, you have the skill to accurately identify the category of social relationships portrayed in an image based on its text description. Your expertise covers 16 distinct types of social relationships, with each pair of individuals falling under one of these 16 categories. Using the provided information, you draw inferences to determine the most likely type of social relationship depicted in an image. Your final output should be one of 16 distinct types of social relationships, defined as follows: {{<father-child>, <mother-child>, <grandpa-grandchild>, <grandma-grandchild>, <friends>, <siblings>, <classmates>, <lovers/spouses>, <presenter-audience>, <teacher-student>, <trainer-trainee>, <leader-subordinate>, <band members>, <dance team members>, <sport team members>, <colleagues>}}.\n\nHere is the definition: \n'
        prompt_prefix_2 = '1. <father-child>: The relationship between a father and his child, referring to a male who becomes the biological or legal father of one or more children.\n'
        prompt_prefix_3 = '2. <mother-child>: The relationship between a mother and her child, referring to a female who becomes the biological or legal mother of one or more children.\n'
        prompt_prefix_4 = '3. <grandpa-grandchild>: The relationship between a grandfather and his grandchild, referring to a male who becomes the grandfather of one or more grandchildren.\n'
        prompt_prefix_5 = '4. <grandma-grandchild>: The relationship between a grandmother and her grandchild, referring to a female who becomes the grandmother of one or more grandchildren.\n'
        prompt_prefix_6 = '5. <friends>: The relationship between two or more individuals who establish an intimate connection, usually based on shared interests, experiences, or backgrounds.\n'
        prompt_prefix_7 = '6. <siblings>: The relationship between two or more individuals who share the same parents or blood relations.\n'
        prompt_prefix_8 = '7. <classmates>: The relationship between students who study in the same class.\n'
        prompt_prefix_9 = '8. <lovers/spouses>: The romantic relationship between two individuals, which may include a marriage relationship.\n'
        prompt_prefix_10 = '9. <presenter-audience>: The relationship between a speaker and a group of listeners, where the speaker (usually a professional) delivers a speech or presentation to the audience, who may be viewers, listeners, spectators, or clients.\n'
        prompt_prefix_11 = '10. <teacher-student>: The relationship between a teacher and one or more students, where the teacher (usually a professional) imparts knowledge, skills, and values to the student.\n'
        prompt_prefix_12 = '11. <trainer-trainee>: The relationship between a trainer and one or more trainees, where the trainer imparts specific knowledge, skills, and techniques.\n'
        prompt_prefix_13 = '12. <leader-subordinate>: The relationship between a leader and their subordinates, where the leader holds a managerial position in an organization or institution, guiding and directing the activities of their subordinates.\n'
        prompt_prefix_14 = '13. <band members>: The relationship between musicians or singers who form a group to perform music together.\n'
        prompt_prefix_15 = '14. <dance team members>: The relationship between dancers who form a group to perform dance routines together.\n'
        prompt_prefix_16 = '15. <sport team members>: The relationship between athletes who form a team to compete in various sports.\n'
        prompt_prefix_17 = '16. <colleagues>: The relationship between individuals who work in the same organization or company.\n'
        prompt_prefix_18 = '\n****************************************************************\n'
        prompt_prefix_19 = 'Example: \n[1. image description]:\nIn a meeting room, two men in suits are engaged in a professional discussion. The first man, <P1>, stands on the left side of the room, wearing a suit with a red tie. He is in his mid-50s and has a warm smile on his face. The second man, <P2>, stands on the right side, wearing a suit and tie. He is in his early 20s and is gesturing with his hands as he speaks. In front of them, there is a red chair, <O2>, which adds a touch of color to the scene. Behind the men, there is a wall with the words "the best way to get your," <O1>, emphasizing the importance of the meeting. On the wall, there is also a red heart, <O3>, symbolizing passion and dedication. Additionally, there is a red beret with black trim, <O4>, placed on a nearby table, adding a touch of style to the room. On the floor, there are a pair of red shoes, <O5>, which complement the overall aesthetic. The combination of these objects creates a visually appealing and professional atmosphere for the meeting. The interaction between the two men, <P1> and <P2>, is evident as they shake hands, symbolizing mutual respect and collaboration. Overall, this scene captures a significant moment of interaction and exchange between two individuals in a professional setting, with the objects enhancing the ambiance and adding depth to the overall composition.\n\n'
        prompt_prefix_20 = '[2. Question]: \nWhat are the most likely social relationships between P1 and P2? Choose only one from {{<father-child>, <mother-child>, <grandpa-grandchild>, <grandma-grandchild>, <friends>, <siblings>, <classmates>, <lovers/spouses>, <presenter-audience>, <teacher-student>, <trainer-trainee>, <leader-subordinate>, <band members>, <dance team members>, <sport team members>, <colleagues>}}..\n\n'
        prompt_prefix_21 = '[3. Answer]: [The final answer is <colleagues>]. Based on the description of the image, the most likely social relationship between P1 and P2 is that they are colleagues. The setting of a meeting room and their professional attire suggest a work-related context. The description emphasizes their professional discussion and the interaction between them, indicating a professional relationship rather than a personal or familial one.\n'
        prompt_prefix_22 = '****************************************************************\n\n'
        prompt_prefix_23 = '[1. image description]:\n'
        prompt_prefix_24 = '\n\n[2. Question]: \nWhat are the most likely social relationships between '
        prompt_prefix_25 = ' and '
        prompt_prefix_26 = '? Choose only one from {{<father-child>, <mother-child>, <grandpa-grandchild>, <grandma-grandchild>, <friends>, <siblings>, <classmates>, <lovers/spouses>, <presenter-audience>, <teacher-student>, <trainer-trainee>, <leader-subordinate>, <band members>, <dance team members>, <sport team members>, <colleagues>}}.\n\n[3. Answer]:'
        template = f"{prompt_prefix_1}{prompt_prefix_2}{prompt_prefix_3}{prompt_prefix_4}{prompt_prefix_5}{prompt_prefix_6}{prompt_prefix_7}{prompt_prefix_8}{prompt_prefix_9}{prompt_prefix_10}{prompt_prefix_11}{prompt_prefix_12}{prompt_prefix_13}{prompt_prefix_14}{prompt_prefix_15}{prompt_prefix_16}{prompt_prefix_17}{prompt_prefix_18}{prompt_prefix_19}{prompt_prefix_20}{prompt_prefix_21}{prompt_prefix_22}{prompt_prefix_23}{{story}}{prompt_prefix_24}{{P1}}{prompt_prefix_25}{{P2}}{prompt_prefix_26}"
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

