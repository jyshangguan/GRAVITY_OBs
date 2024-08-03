from astropy.io.votable.tree import VOTableFile, Resource, Field , CooSys
from astropy.io.votable.tree import Table as voTable 
from astropy.table import Table, vstack

def CreateAsproVoTableFile(filename: str, dataset: Table, col_name='Name', col_ra='ra', col_dec='dec') -> None :
        '''
        Create a VOTable file for Aspro2
        '''

        # Create a new VOTable file with version 1.1 standards
        votable = VOTableFile(version='1.1')

        #Add Aspro2 Coordinate System
        coord_system = CooSys(ID='J2000', equinox='2000', epoch='J2000', system='eq_FK5')
        votable.coordinate_systems.append(coord_system)

        #Add one resource
        #resource = Resource( ID='aaa' , name='aspro2')
        resource = Resource(name='aspro2', ID='AAA', utype=None, type='results')
        votable.resources.append(resource)
        # ... with one table
        table = voTable(votable, name="Targets")
        resource.tables.append(table)

        #Define the fields you want for your file
        table.fields.extend([
                Field(votable,  name ='NAME'  , datatype="char" , ucd="meta.id;meta.main"   , arraysize="*" )                             ,
                Field(votable,  name="RA"     , datatype="char" , ucd="pos.eq.ra;meta.main" , arraysize="*" , unit="h:m:s" , ref="J2000") ,
                Field(votable,  name='DEC'    , datatype="char" , ucd="pos.eq.dec;meta.main" , arraysize="*" , unit="d:m:s" , ref="J2000")   ])

        #Number of entries
        ntargets = len(dataset) 
        table.create_arrays(ntargets)

        #For each entry create a table
        for (idx,target) in enumerate(dataset):
                table.array[idx] = (target[col_name], target[col_ra] , target[col_dec] )

        #Export xml file
        votable.to_xml(filename)