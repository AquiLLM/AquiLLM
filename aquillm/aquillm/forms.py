from django import forms
from .models import Collection, CollectionPermission
from django.contrib.auth import get_user_model

User = get_user_model()


class UserCollectionMultipleChoiceField(forms.ModelMultipleChoiceField):
    def __init__(self, user, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user
        self.queryset = self._get_queryset()
        self.initial = self._get_queryset()

    def _get_queryset(self):
        return Collection.objects.filter(
            pk__in=[perm.collection.id for perm in CollectionPermission.objects.filter(user=self.user).distinct()]
        )
    
class UserCollectionSingleChoiceField(forms.ModelChoiceField):
    def __init__(self, user, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.user = user
        self.queryset = self._get_queryset()

    def _get_queryset(self):
        return Collection.objects.filter(
            pk__in=[perm.collection.id for perm in CollectionPermission.objects.filter(user=self.user).distinct()]
        )
    

class NewCollectionForm(forms.Form):
    name = forms.CharField(
        max_length=255,
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    viewers = forms.ModelMultipleChoiceField(
        queryset=User.objects.all(),
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-control'}),
    )
    
    editors = forms.ModelMultipleChoiceField(
        queryset=User.objects.all(),
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-control'}),
    )
    
    admins = forms.ModelMultipleChoiceField(
        queryset=User.objects.filter(),
        required=False,
        widget=forms.CheckboxSelectMultiple(attrs={'class': 'form-control'}),
    )
    
    def __init__(self, *args, **kwargs):
        user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)

        if user:
            qs = User.objects.exclude(id=user.id)
            self.fields['viewers'].queryset = qs
            self.fields['editors'].queryset = qs
            self.fields['admins'].queryset = qs

    def clean(self):
        cleaned_data = super().clean()
        cleaned_data['editors'] = cleaned_data['editors'].exclude(pk__in=cleaned_data['admins'])
        cleaned_data['viewers'] = cleaned_data['viewers'].exclude(pk__in=cleaned_data['editors'])
        cleaned_data['viewers'] = cleaned_data['viewers'].exclude(pk__in=cleaned_data['admins'])

        return cleaned_data

class SearchForm(forms.Form):
    def __init__(self, user, *args, **kwargs):
        super().__init__(*args, **kwargs)

        collections_attrs = {
            'class': 'rounded-lg bg-scheme-shade_3 mb-4 max-h-[300px] overflow-y-auto px-4 py-2 border border-border-high_contrast',
        }
        self.fields['collections'] = UserCollectionMultipleChoiceField(
            user=user,
            widget=forms.CheckboxSelectMultiple(attrs=collections_attrs),
            queryset=Collection.objects.none(), # this is weird but necessary
            required=False,
        )

    query_attrs = {
        'class': 'w-full h-full resize-none p-3 mb-4 rounded-md bg-scheme-shade_3 placeholder:text-text-less_contrast text-wrap text-text-normal',
        'rows': 4,
        'placeholder': 'Send a message'}
    
    query = forms.CharField(label="Query", widget=forms.Textarea(attrs=query_attrs), max_length=10000)

    top_k_attrs = {
        'class': 'w-full p-2 rounded-md bg-scheme-shade_3 placeholder:text-text-low_contrast text-text-normal mb-4',
    }

    top_k = forms.IntegerField(widget=forms.NumberInput(attrs=top_k_attrs), min_value=1, max_value=200, initial=5)


class ArXiVForm(forms.Form):
    arxiv_id = forms.CharField(
        label="Article arXiv Identifier",  
        widget=forms.TextInput(attrs={
            'placeholder': 'Article arXiv Identifier:', 
            'class': 'rounded-[8px] w-full p-2 bg-scheme-shade_4 border border-border-high_contrast placeholder:gray-shade_9' 
        })
    )

    # collections_attrs = {
    #     'class': 'rounded-md bg-lightest-primary max-h-[200px] overflow-y-auto bg-scheme-shade_3 border-border-high_contrast',
    # }

    # collection = forms.ChoiceField(widget=forms.RadioSelect(attrs=collections_attrs))

    # def __init__(self, user, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.fields['collection'].choices = [(col.id, col.name) for col in Collection.objects.filter_by_user_perm(user, perm="EDIT")]

    def __init__(self, user, *args, **kwargs):
        super().__init__(*args, **kwargs)
        collections_attrs = {
           'class': 'rounded-lg max-h-[200px] overflow-y-auto bg-scheme-shade_4 border border-border-high_contrast',
        }
        self.fields['collection'] = UserCollectionSingleChoiceField(
            user=user,
            widget=forms.RadioSelect(attrs=collections_attrs),
            queryset=Collection.objects.none(), # this is weird but necessary
            required=True,
        )


class PDFDocumentForm(forms.Form):
    title = forms.CharField(
        label="Article Title",  
        widget=forms.TextInput(attrs={
            'placeholder': 'Article Title:', 
            'class': 'rounded-[8px] w-full p-2 bg-scheme-shade_4 border border-border-high_contrast placeholder:gray-shade_9' 
        })
    )

    pdf_file = forms.FileField(
        label="PDF File",
        widget=forms.ClearableFileInput(attrs={
            'class': 'hidden',  # hide the default file input
            'id': 'pdf-file-input'  # assign an ID for linking the custom label
        })
    )

    # TODO: make function sig not weird
    def __init__(self, user, *args, **kwargs):
        super().__init__(*args, **kwargs)
        collections_attrs = {
           'class': 'rounded-lg max-h-[200px] overflow-y-auto bg-scheme-shade_4 border border-border-high_contrast',
        }
        self.fields['collection'] = UserCollectionSingleChoiceField(
            user=user,
            widget=forms.RadioSelect(attrs=collections_attrs),
            queryset=Collection.objects.none(), # this is weird but necessary
            required=True,
        )
    
class VTTDocumentForm(forms.Form):
    title = forms.CharField(
        label="Transcript Title",  
        widget=forms.TextInput(attrs={
            'placeholder': 'Transcript Title:', 
            'class': 'rounded-[8px] w-full p-2 bg-scheme-shade_4 border border-border-high_contrast placeholder:gray-shade_9' 
        })
    )

    vtt_file = forms.FileField(
        label="Audio File (optional)",
        required=False,
        widget=forms.ClearableFileInput(attrs={
            'class': 'hidden',
            'id': 'audio-file-input'
        })
    )

    vtt_file = forms.FileField(
        label=".vtt Transcript File",
        widget=forms.ClearableFileInput(attrs={
            'class': 'hidden',
            'id': 'vtt-file-input'
        })
    )

    def __init__(self, user, *args, **kwargs):
        super().__init__(*args, **kwargs)
        collections_attrs = {
           'class': 'rounded-lg max-h-[200px] overflow-y-auto bg-scheme-shade_4 border border-border-high_contrast',
        }
        self.fields['collection'] = UserCollectionSingleChoiceField(
            user=user,
            widget=forms.RadioSelect(attrs=collections_attrs),
            queryset=Collection.objects.none(), # this is weird but necessary
            required=True,
        )

class HandwrittenNotesForm(forms.Form):
    title = forms.CharField(
        label="Handwritten Notes Title",  
        widget=forms.TextInput(attrs={
            'placeholder': 'Notes Title:', 
            'class': 'rounded-[8px] w-full p-2 bg-scheme-shade_4 border border-border-high_contrast placeholder:gray-shade_9' 
        })
    )
    
    image_file = forms.ImageField(
        label="Image File",
        widget=forms.ClearableFileInput(attrs={
            'class': 'hidden',  # hide the default file input
            'id': 'image-file-input'  # assign an ID for linking the custom label
        })
    )
    
    # Optional LaTeX conversion checkbox
    convert_to_latex = forms.BooleanField(
        label="Convert to LaTeX",
        required=False,
        widget=forms.CheckboxInput(attrs={
            'class': 'mr-2'
        })
    )

    def __init__(self, user, *args, **kwargs):
        super().__init__(*args, **kwargs)
        collections_attrs = {
           'class': 'rounded-lg max-h-[200px] overflow-y-auto bg-scheme-shade_4 border border-border-high_contrast',
        }
        
        self.fields['collection'] = UserCollectionSingleChoiceField(
            user=user,
            widget=forms.RadioSelect(attrs=collections_attrs),
            queryset=Collection.objects.none(),  # this is weird but necessary
            required=True,
        )
