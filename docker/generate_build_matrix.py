# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Helper script to generate the ``build_matrix.yaml`` and update ``README.md``.

Note: this script requires tabulate. Run ``pip install tabulate`` if not installed

To run::

    python generate_build_matrix.py
"""

import itertools
import os
import sys
from typing import Optional

import packaging.version
import tabulate
import yaml

PRODUCTION_PYTHON_VERSION = '3.12'
PRODUCTION_PYTORCH_VERSION = '2.6.0'
EFA_INSTALLER_VERSION = '1.39.0'
PRODUCTION_UBUNTU_VERSION = '22.04'


def _get_torchvision_version(pytorch_version: str):
    if pytorch_version == '2.6.0':
        return '0.21.0'
    if pytorch_version == '2.5.1':
        return '0.20.1'
    if pytorch_version == '2.4.1':
        return '0.19.1'
    raise ValueError(f'Invalid pytorch_version: {pytorch_version}')


def _version_geq(v1: str, v2: str):
    return packaging.version.parse(v1) >= packaging.version.parse(v2)


def _get_base_image(cuda_version: str, ubuntu_version: str = '22.04'):
    if not cuda_version:
        return f'ubuntu:{ubuntu_version}'
    if _version_geq(cuda_version, '12.2.0'):
        return f'nvidia/cuda:{cuda_version}-cudnn-devel-ubuntu{ubuntu_version}'
    return f'nvidia/cuda:{cuda_version}-cudnn8-devel-ubuntu{ubuntu_version}'


def _get_cuda_version(pytorch_version: str, use_cuda: bool, cuda_variant: Optional[str] = ''):
    # From https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/
    if not use_cuda:
        return ''
    if cuda_variant:
        return cuda_variant
    if pytorch_version == '2.6.0':
        return '12.4.1'
    if pytorch_version == '2.5.1':
        return '12.4.1'
    if pytorch_version == '2.4.1':
        return '12.4.1'
    raise ValueError(f'Invalid pytorch_version: {pytorch_version}')


def _get_cuda_version_tag(cuda_version: str):
    if not cuda_version:
        return 'cpu'
    return 'cu' + ''.join(cuda_version.split('.')[:2])


def _get_cuda_override(cuda_version: str):
    if cuda_version == '12.1.1':
        cuda_121_override_string = (
            'cuda>=12.1 brand=tesla,driver>=450,driver<451 '
            'brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 '
            'brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 '
            'brand=geforce,driver>=470,driver<471 brand=geforcertx,driver>=470,driver<471 '
            'brand=quadro,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 '
            'brand=titan,driver>=470,driver<471 brand=titanrtx,driver>=470,driver<471 '
            'brand=tesla,driver>=510,driver<511 brand=unknown,driver>=510,driver<511 '
            'brand=nvidia,driver>=510,driver<511 brand=nvidiartx,driver>=510,driver<511 '
            'brand=geforce,driver>=510,driver<511 brand=geforcertx,driver>=510,driver<511 '
            'brand=quadro,driver>=510,driver<511 brand=quadrortx,driver>=510,driver<511 '
            'brand=titan,driver>=510,driver<511 brand=titanrtx,driver>=510,driver<511 '
            'brand=tesla,driver>=515,driver<516 brand=unknown,driver>=515,driver<516 '
            'brand=nvidia,driver>=515,driver<516 brand=nvidiartx,driver>=515,driver<516 '
            'brand=geforce,driver>=515,driver<516 brand=geforcertx,driver>=515,driver<516 '
            'brand=quadro,driver>=515,driver<516 brand=quadrortx,driver>=515,driver<516 '
            'brand=titan,driver>=515,driver<516 brand=titanrtx,driver>=515,driver<516 '
            'brand=tesla,driver>=525,driver<526 brand=unknown,driver>=525,driver<526 '
            'brand=nvidia,driver>=525,driver<526 brand=nvidiartx,driver>=525,driver<526 '
            'brand=geforce,driver>=525,driver<526 brand=geforcertx,driver>=525,driver<526 '
            'brand=quadro,driver>=525,driver<526 brand=quadrortx,driver>=525,driver<526 '
            'brand=titan,driver>=525,driver<526 brand=titanrtx,driver>=525,driver<526'
        )

        return cuda_121_override_string
    elif cuda_version == '11.8.0':
        cuda_118_override_string = (
            'cuda>=11.8 brand=tesla,driver>=470,driver<471 '
            'brand=tesla,driver>=515,driver<516 brand=unknown,driver>=470,driver<471 '
            'brand=unknown,driver>=515,driver<516 brand=nvidia,driver>=470,driver<471 '
            'brand=nvidia,driver>=515,driver<516 brand=nvidiartx,driver>=470,driver<471 '
            'brand=nvidiartx,driver>=515,driver<516 brand=geforce,driver>=470,driver<471 '
            'brand=geforce,driver>=515,driver<516 brand=quadro,driver>=470,driver<471 '
            'brand=quadro,driver>=515,driver<516 brand=titan,driver>=470,driver<471 '
            'brand=titan,driver>=515,driver<516 brand=titanrtx,driver>=470,driver<471 '
            'brand=titanrtx,driver>=515,driver<516'
        )
        return cuda_118_override_string
    return ''


def _get_pytorch_tags(
    python_version: str,
    pytorch_version: str,
    cuda_version: str,
    stage: str,
    interconnect: str,
    ubuntu_version: str,
):
    if stage == 'pytorch_stage':
        base_image_name = 'mosaicml/pytorch'
    else:
        raise ValueError(f'Invalid stage: {stage}')
    tags = []
    cuda_version_tag = _get_cuda_version_tag(cuda_version)
    tags += [
        f'{base_image_name}:{pytorch_version}_{cuda_version_tag}-python{python_version}-ubuntu{ubuntu_version}',
    ]

    if python_version == PRODUCTION_PYTHON_VERSION and pytorch_version == PRODUCTION_PYTORCH_VERSION and ubuntu_version == PRODUCTION_UBUNTU_VERSION:
        if not cuda_version:
            tags += [f'{base_image_name}:latest_cpu']
        else:
            tags += [f'{base_image_name}:latest']

    if interconnect == 'EFA':
        tags = [f'{tag}-aws' for tag in tags]
    return tags


def _get_composer_tags(composer_version: str, use_cuda: bool):
    base_image_name = 'mosaicml/composer'

    tags = []
    if not use_cuda:
        tags += [f'{base_image_name}:{composer_version}_cpu']
        tags += [f'{base_image_name}:latest_cpu']
    else:
        tags += [f'{base_image_name}:{composer_version}']
        tags += [f'{base_image_name}:latest']
    print(tags)
    return tags


def _get_image_name(pytorch_version: str, cuda_version: str, stage: str, interconnect: str, ubuntu_version: str):
    pytorch_version = pytorch_version.replace('.', '-')
    cuda_version = _get_cuda_version_tag(cuda_version)

    if stage == 'pytorch_stage':
        stage = ''
    else:
        raise ValueError(f'Invalid stage: {stage}')

    if ubuntu_version != PRODUCTION_UBUNTU_VERSION:
        base_os = '-ub' + ubuntu_version.replace('.', '')
    else:
        base_os = ''

    if interconnect == 'EFA':
        fabric = '-aws'
    else:
        fabric = ''

    return f'torch{stage}-{pytorch_version}-{cuda_version}{base_os}{fabric}'


def _write_table(table_tag: str, table_contents: str):
    with open(os.path.join(os.path.dirname(__name__), 'README.md'), 'r') as f:
        contents = f.read()

    begin_table_tag = f'<!-- BEGIN_{table_tag} -->'
    end_table_tag = f'<!-- END_{table_tag} -->'

    pre = contents.split(begin_table_tag)[0]
    if end_table_tag in contents:
        post = contents.split(end_table_tag)[1]
    else:
        print(f"Warning: '{end_table_tag}' not found in contents.")
        post = ''
    new_readme = f'{pre}{begin_table_tag}\n{table_contents}\n{end_table_tag}{post}'

    with open(os.path.join(os.path.dirname(__name__), 'README.md'), 'w') as f:
        f.write(new_readme)


def _cross_product_extra_cuda(
    python_pytorch_versions: list,
    pytorch_cuda_variants_extra: dict,
    cuda_options: list,
    *args,
):
    for product in itertools.product(python_pytorch_versions, cuda_options, *args):
        (python_version, pytorch_version), use_cuda, *rest = product
        cuda_variants = ['']
        if use_cuda and pytorch_version in pytorch_cuda_variants_extra:
            cuda_variants.extend(pytorch_cuda_variants_extra[pytorch_version])
        for cuda_variant in cuda_variants:
            yield (python_version, pytorch_version), use_cuda, cuda_variant, *rest


def _main():
    python_pytorch_versions = [('3.12', '2.6.0'), ('3.12', '2.5.1'), ('3.12', '2.4.1')]
    pytorch_cuda_variants_extra = {
        '2.6.0': ['12.6.3'],
    }  # Extra cuda variants to be built in addition to the defaults
    cuda_options = [True, False]
    stages = ['pytorch_stage']
    interconnects = ['mellanox', 'EFA']  # mellanox is default, EFA needed for AWS
    ubuntu_versions = ['22.04']

    pytorch_entries = []

    pytorch_products = _cross_product_extra_cuda(
        python_pytorch_versions,
        pytorch_cuda_variants_extra,
        cuda_options,
        stages,
        interconnects,
        ubuntu_versions,
    )
    # Add a couple of entries for legacy platforms (Python 3.11, Ubuntu 20.04, no special interconnect)
    legacy_pytorch_products = [
        (('3.11', '2.6.0'), True, '', 'pytorch_stage', '', '20.04'),
        (('3.11', '2.6.0'), True, '12.6.3', 'pytorch_stage', '', '20.04'),
    ]

    for product in itertools.chain(pytorch_products, legacy_pytorch_products):
        (python_version, pytorch_version), use_cuda, cuda_variant, stage, interconnect, ubuntu_version = product

        cuda_version = _get_cuda_version(pytorch_version=pytorch_version, use_cuda=use_cuda, cuda_variant=cuda_variant)

        entry = {
            'IMAGE_NAME':
                _get_image_name(pytorch_version, cuda_version, stage, interconnect, ubuntu_version),
            'BASE_IMAGE':
                _get_base_image(cuda_version, ubuntu_version),
            'CUDA_VERSION':
                cuda_version,
            'PYTHON_VERSION':
                python_version,
            'PYTORCH_VERSION':
                pytorch_version,
            'UBUNTU_VERSION':
                ubuntu_version,
            'TARGET':
                stage,
            'TORCHVISION_VERSION':
                _get_torchvision_version(pytorch_version),
            'TAGS':
                _get_pytorch_tags(
                    python_version=python_version,
                    pytorch_version=pytorch_version,
                    cuda_version=cuda_version,
                    stage=stage,
                    interconnect=interconnect,
                    ubuntu_version=ubuntu_version,
                ),
            'PYTORCH_NIGHTLY_URL':
                '',
            'PYTORCH_NIGHTLY_VERSION':
                '',
            'NVIDIA_REQUIRE_CUDA_OVERRIDE':
                _get_cuda_override(cuda_version),
        }

        # Only build EFA image on cuda and pytorch_stage
        if interconnect == 'EFA' and not (use_cuda and stage == 'pytorch_stage'):
            continue

        # Skip the mellanox drivers if not required or not in the cuda images
        if not cuda_version or interconnect != 'mellanox':
            entry['MOFED_VERSION'] = ''
        else:
            entry['MOFED_VERSION'] = 'latest-23.10'

        # Skip EFA drivers if not using EFA
        if interconnect != 'EFA':
            entry['EFA_INSTALLER_VERSION'] = ''
        else:
            entry['EFA_INSTALLER_VERSION'] = EFA_INSTALLER_VERSION

        pytorch_entries.append(entry)

    composer_entries = []

    # The `GIT_COMMIT` is a placeholder and Jenkins will substitute it with the actual git commit for the `composer_staging` images
    composer_versions = ['0.30.0']  # Only build images for the latest composer version
    composer_python_versions = [PRODUCTION_PYTHON_VERSION]  # just build composer against the latest

    for product in itertools.product(composer_python_versions, composer_versions, cuda_options):
        python_version, composer_version, use_cuda = product

        pytorch_version = PRODUCTION_PYTORCH_VERSION
        cuda_version = _get_cuda_version(pytorch_version=pytorch_version, use_cuda=use_cuda)
        cpu = '-cpu' if not use_cuda else ''

        entry = {
            'IMAGE_NAME': f"composer-{composer_version.replace('.', '-')}{cpu}",
            'BASE_IMAGE': _get_base_image(cuda_version),
            'CUDA_VERSION': cuda_version,
            'PYTHON_VERSION': python_version,
            'PYTORCH_VERSION': pytorch_version,
            'UBUNTU_VERSION': PRODUCTION_UBUNTU_VERSION,
            'PYTORCH_NIGHTLY_URL': '',
            'PYTORCH_NIGHTLY_VERSION': '',
            'TARGET': 'composer_stage',
            'TORCHVISION_VERSION': _get_torchvision_version(pytorch_version),
            'MOFED_VERSION': 'latest-23.10',
            'EFA_INSTALLER_VERSION': '',
            'COMPOSER_INSTALL_COMMAND': f'mosaicml[all]=={composer_version}',
            'TAGS': _get_composer_tags(
                composer_version=composer_version,
                use_cuda=use_cuda,
            ),
            'NVIDIA_REQUIRE_CUDA_OVERRIDE': _get_cuda_override(cuda_version),
        }

        composer_entries.append(entry)

    with open(os.path.join(os.path.dirname(__name__), 'build_matrix.yaml'), 'w+') as f:
        f.write('# This file is automatically generated by generate_build_matrix.py. DO NOT EDIT!\n')
        yaml.safe_dump(pytorch_entries + composer_entries, f)

    # Also print the table for the readme

    # PyTorch Table
    headers = ['Linux Distro', 'Flavor', 'PyTorch Version', 'CUDA Version', 'Python Version', 'Docker Tags']
    table = []
    for entry in pytorch_entries:
        interconnect = 'N/A'
        if entry['CUDA_VERSION']:
            if entry['MOFED_VERSION'] != '':
                interconnect = 'Infiniband'
            elif entry['EFA_INSTALLER_VERSION'] != '':
                interconnect = 'EFA'
        cuda_version = f"{entry['CUDA_VERSION']} ({interconnect})" if entry['CUDA_VERSION'] else 'cpu'
        linux_distro = f"Ubuntu {entry['UBUNTU_VERSION']}"
        table.append([
            linux_distro,
            'Base',  # Flavor
            entry['PYTORCH_VERSION'],  # Pytorch version
            cuda_version,  # Cuda version
            entry['PYTHON_VERSION'],  # Python version,
            ', '.join(reversed([f'`{x}`' for x in entry['TAGS']])),  # Docker tags
        ])
    table.sort(
        key=lambda x: x[3].replace('Infiniband', '1').replace('EFA', '2'),
    )  # cuda version, put infiniband ahead of EFA
    table.sort(key=lambda x: packaging.version.parse(x[4]), reverse=True)  # python version
    table.sort(key=lambda x: packaging.version.parse(x[2]), reverse=True)  # pytorch version
    table.sort(key=lambda x: x[1])  # flavor
    table_contents = tabulate.tabulate(table, headers, tablefmt='github', floatfmt='', disable_numparse=True)
    _write_table('PYTORCH_BUILD_MATRIX', table_contents)

    # Composer Table
    # Also print the table for the readme
    headers = ['Composer Version', 'CUDA Support', 'Docker Tag']
    table = []
    for entry in composer_entries:

        if len(entry['TAGS']) == 0:
            continue

        table.append([
            entry['TAGS'][0].split(':')[1].replace('_cpu', ''),  # Composer version, or 'latest'
            'No' if entry['BASE_IMAGE'].startswith('ubuntu:') else 'Yes',  # Whether there is Cuda support
            ', '.join(reversed([f'`{x}`' for x in entry['TAGS']])),  # Docker tags
        ])
    table.sort(key=lambda x: x[1], reverse=True)  # cuda support
    table.sort(
        key=lambda x: packaging.version.parse('9999999999999' if x[0] == 'latest' else x[0]),
        reverse=True,
    )  # Composer version
    table_contents = tabulate.tabulate(table, headers, tablefmt='github', floatfmt='', disable_numparse=True)
    _write_table('COMPOSER_BUILD_MATRIX', table_contents)

    print('Successfully updated `build_matrix.yaml` and `README.md`.', file=sys.stderr)


if __name__ == '__main__':
    _main()
