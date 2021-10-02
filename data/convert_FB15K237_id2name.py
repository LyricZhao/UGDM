# Read id-to-entity mapping
id2entity = {}
with open('FB15K237_entity2id.txt') as file:
    n = int(file.readline())
    for i in range(n):
        content = file.readline()
        entity, id = content.strip().split()
        id2entity[int(id)] = entity

# Read entity-to-name mapping
entity2name = {}
with open('FB_entity2name.tsv') as file:
    for content in file.readlines():
        content = content.strip().split()
        entity, name = content[0], ' '.join(content[1:])
        if not entity in entity2name:
            entity2name[entity] = name
        else:
            entity2name[entity] += ', ' + name

# Write id-to-name mapping
with open('FB15K237_id2name.txt', 'w') as file:
    file.write('{}\n'.format(len(id2entity)))
    for id, entity in id2entity.items():
        if not entity in entity2name:
            print('Warning: entity {} not in mapping'.format(entity))
            entity2name[entity] = 'Unknown'
        file.write('{} {}\n'.format(id, entity2name[entity]))
