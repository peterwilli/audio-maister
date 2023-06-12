import setuptools
setuptools.setup(
    name='audio-maister',
    version='1.0',
    scripts=['./audiomaister/audiomaister'],
    author='Peter Willemsen <peter@codebuffet.co>',
    description='General purpose audio restoration AI',
    packages=['audiomaister'],
    license="GNU Affero General Public License v3.0",
    install_requires=[
        "GitPython",
        "numpy",
        "soundfile",
        "scipy",
        "librosa==0.8.1",
        "matplotlib",
        "torchlibrosa==0.0.7",
        "tensorboard",
        "tqdm",
        "datasets",
        "torch",
        "lightning"
    ],
    python_requires='>=3.5'
)
