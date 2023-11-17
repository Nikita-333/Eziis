import os

from django.core.exceptions import ValidationError
from django.db import models

# Create your models here.
class FileProject(models.Model):
    file = models.FileField(upload_to='uploads')

    def __str__(self):
        return str(self.file)