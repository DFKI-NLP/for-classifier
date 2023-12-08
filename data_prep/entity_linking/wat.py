import json
import requests as req


# Code taken from the WAT API documentation: https://sobigdata.d4science.org/web/tagme/wat-api
class WATAnnotation:
    # An entity annotated by WAT

    def __init__(self, d):
        # char offset (included)
        self.start = d['start']
        # char offset (not included)
        self.end = d['end']

        # annotation accuracy
        self.rho = d['rho']
        # spot-entity probability
        self.prior_prob = d['explanation']['prior_explanation']['entity_mention_probability']

        # annotated text
        self.spot = d['spot']

        # Wikpedia entity info
        self.wiki_id = d['id']
        self.wiki_title = d['title']

    def json_dict(self):
        # Simple dictionary representation
        return {'wiki_title': self.wiki_title,
                'wiki_id': self.wiki_id,
                'start': self.start,
                'end': self.end,
                'rho': self.rho,
                'prior_prob': self.prior_prob
                }


def wat_entity_linking(text):
    # Main method, text annotation with WAT entity linking system
    wat_url = 'https://wat.d4science.org/wat/tag/tag'
    MY_GCUBE_TOKEN = '4be5b632-2c72-4875-a941-5ac12e4c9a44-843339462'
    payload = [("gcube-token", MY_GCUBE_TOKEN),
               ("text", text),
               ("lang", 'en'),
               ("tokenizer", "nlp4j"),
               ('debug', 9),
               ("method",
                "spotter:includeUserHint=true:includeNamedEntity=true:includeNounPhrase=true,prior:k=50,filter-valid,centroid:rescore=true,topk:k=10,voting:relatedness=lm,ranker:model=0046.model,confidence:model=pruner-wiki.linear")]

    response = req.get(wat_url, params=payload)
    return [WATAnnotation(a) for a in response.json()['annotations']]


def get_wat_annotations(wat_annotations):
    json_list = [w.json_dict() for w in wat_annotations]
    return json.dumps(json_list, indent=4)
