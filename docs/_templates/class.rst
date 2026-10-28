{{ objname | escape | underline }}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :show-inheritance:

   {% block methods %}
   {% set public = methods | reject("equalto", "__init__") | list %}
   {% if public %}
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   {% for item in public %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}


   .. rubric:: Details
