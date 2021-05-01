from setuptools import setup

def parse_requirements(requirements):
      with open(requirements) as f:
            return [l.strip('\n') for l in f if l.strip('\n') and not l.startswith('#')]

reqs = parse_requirements('requirements.txt')
print(reqs)

setup(name='lords_ai',
      version='0.0.1',
      install_requires=reqs
      )