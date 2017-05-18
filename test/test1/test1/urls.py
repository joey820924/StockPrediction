from django.conf.urls import patterns, include, url
from views import here
from views import add
from views import getPrice
from django.contrib import admin
admin.autodiscover()

urlpatterns = patterns('',
    # Examples:
    # url(r'^$', 'test1.views.home', name='home'),
    # url(r'^blog/', include('blog.urls')),

    url(r'^admin/', include(admin.site.urls)),
    url(r'^here/$',here),
    url(r'^(\d{1,2})/plus/(\d{1,2})/$', add),
    #url(r'^([-+]?[0-9]+.[0-9]*)/([-+]?[0-9]+.[0-9]*)/([-+]?[0-9]+.[0-9]*)/([-+]?[0-9]+.[0-9]*)/([-+]?[0-9]+.[0-9]*)/([-+]?[0-9]+.[0-9]*)/([-+]?[0-9]+.[0-9]*)/([-+]?[0-9]+.[0-9]*)/([-+]?[0-9]+.[0-9]*)/$', getPrice),
    url(r'^user/(?P<username>\w{1,50})/$',getPrice)
)