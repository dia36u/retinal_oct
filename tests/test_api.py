import unittest
import requests


class APITest(unittest.TestCase):
    def test_classification_route(self):
        # Préparation des données de test
        image_path = "static/img.jpeg"
        url = "http://localhost:5000/predict"
        prediction = '1'
        # Envoi de la requête POST avec l'image
        files = {'file': open(image_path, 'rb')}
        response = requests.post(url, files=files)
        # Vérification de la réponse
        # Vérifie que la requête s'est bien déroulée
        self.assertEqual(response.status_code, 200)
        result = response.json()
        # Vérifie que le résultat contient 1
        self.assertIn(prediction, str(result))


if __name__ == '__main__':
    unittest.main()
