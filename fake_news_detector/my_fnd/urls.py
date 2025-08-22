from django.urls import path

from . import views

urlpatterns = [ 
    path('', views.index, name='index'),
    path('newpage/',  views.new_page,  name="my_function"),
    path('ifram/', views.ifram ,name="ifram")
    

]