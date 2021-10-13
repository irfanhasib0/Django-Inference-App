from django.db import models

# Create your models here.
class DataIO(models.Model):
    index = models.IntegerField(primary_key=True)
    col_1 = models.FloatField()
    col_2 = models.FloatField()
    col_3 = models.FloatField()
    col_4 = models.FloatField()
    
    #def __str__(self):
    #    return self.title
    
    
# Create your models here.
class ImageIO(models.Model):
    index = models.IntegerField(primary_key=True)
    col_1 = models.CharField(max_length=150)
    col_2 = models.CharField(max_length=150)
    #def __str__(self):
    #    return self.title
    
        